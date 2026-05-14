use ash::prelude::VkResult;
use ash::vk;

use crate::devices::Device;
use crate::swap_chain::SwapChain;

#[non_exhaustive]
pub struct SceneSync {
    pub present_complete_semaphores: Vec<vk::Semaphore>,
    pub render_finished_semaphores: Vec<vk::Semaphore>,
    pub inflight_fences: Vec<vk::Fence>,
}

impl SceneSync {
    pub fn new(
        device: &Device,
        swap_chain: &SwapChain,
        max_frames_inflight: usize,
    ) -> VkResult<Self> {
        let device_h = &device.handle;

        let semaphore_ci = vk::SemaphoreCreateInfo::default();
        let render_finished_semaphores =
            (0..swap_chain.images.len()) // indexed by swapchain image
                .map(|_| unsafe { device_h.create_semaphore(&semaphore_ci, None) })
                .collect::<Result<Vec<_>, _>>()?;
        let present_complete_semaphores =
            (0..max_frames_inflight) // indexed by frame slot
                .map(|_| unsafe { device_h.create_semaphore(&semaphore_ci, None) })
                .collect::<Result<Vec<_>, _>>()?;

        let fence_ci = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        let inflight_fences =
            (0..max_frames_inflight) // indexed by frame slot
                .map(|_| unsafe { device_h.create_fence(&fence_ci, None) })
                .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            present_complete_semaphores,
            render_finished_semaphores,
            inflight_fences,
        })
    }

    /// # Safety
    ///
    /// - Must be called before the `ash::Device` that was used to create this
    ///   `SyncObjects` is destroyed.
    /// - All semaphores and fences must not be in a pending state (i.e. no
    ///   GPU operations are waiting on or signalling them).
    /// - Must be called at most once. Calling it more than once is undefined
    ///   behaviour as the underlying handles become invalid after the first call.
    pub unsafe fn destroy(&mut self, device: &Device) {
        let device_h = &device.handle;
        unsafe {
            self.present_complete_semaphores
                .iter()
                .for_each(|&s| device_h.destroy_semaphore(s, None));
            self.render_finished_semaphores
                .iter()
                .for_each(|&s| device_h.destroy_semaphore(s, None));
            self.inflight_fences
                .iter()
                .for_each(|&f| device_h.destroy_fence(f, None));
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////

/// Synchronization primitives for the particles render loop.
///
/// # Timeline semaphore protocol
///
/// Unlike binary semaphores, a timeline semaphore carries a monotonically
/// increasing `u64` counter. Instead of being "signaled" or "unsignaled", it
/// holds a value, and waiters block until the counter reaches a specific target
/// value.
///
/// Each call to `draw_frame` consumes two timeline values:
///
/// ```text
/// timeline_value (before frame):  N
///
/// compute submit:  wait for N,    signal N+1
/// graphics submit: wait for N+1,  signal N+2
///
/// timeline_value (after frame):   N+2
/// ```
///
/// This creates an implicit dependency chain:
/// - The compute shader waits until the previous frame's graphics work has
///   signaled (or the initial value 0 on the first frame), ensuring it does
///   not overwrite the storage buffer while the GPU is still reading it for
///   rendering.
/// - The graphics shader waits until compute has signaled, ensuring it reads
///   the particle positions only after the compute pass has finished writing
///   them.
///
/// The CPU waits on `graphics_signal_value` via `wait_semaphores` before
/// calling `queue_present`, ensuring the swapchain image is fully rendered
/// before it is handed back to the compositor.
///
/// The in-flight fence is used separately: it is signaled by
/// `acquire_next_image` and waited on at the start of the frame to ensure
/// we do not re-record command buffers that the GPU may still be executing
/// from a previous frame using the same `frame_index` slot.
#[non_exhaustive]
pub struct ParticlesSync {
    pub semaphore: vk::Semaphore,
    pub timeline_value: u64,
    pub inflight_fences: Vec<vk::Fence>,
}

impl ParticlesSync {
    pub fn new(device: &Device, max_frames_inflight: usize) -> VkResult<Self> {
        let mut semaphore_type_ci = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let semaphore_ci = vk::SemaphoreCreateInfo::default().push_next(&mut semaphore_type_ci);
        let semaphore = unsafe { device.handle.create_semaphore(&semaphore_ci, None)? };

        let fence_ci = vk::FenceCreateInfo::default();
        let inflight_fences = (0..max_frames_inflight)
            .map(|_| unsafe { device.handle.create_fence(&fence_ci, None) })
            .collect::<VkResult<Vec<_>>>()?;

        Ok(Self {
            semaphore,
            timeline_value: 0,
            inflight_fences,
        })
    }

    /// # Safety
    ///
    /// - Must be called before the `ash::Device` that was used to create this
    ///   `ParticlesSync` is destroyed.
    /// - All semaphores and fences must no longer be in use by the GPU.
    /// - Must be called at most once.
    pub unsafe fn destroy(&mut self, device: &Device) {
        unsafe {
            device.handle.destroy_semaphore(self.semaphore, None);
            for fence in self.inflight_fences.drain(..) {
                device.handle.destroy_fence(fence, None);
            }
        }
    }
}
