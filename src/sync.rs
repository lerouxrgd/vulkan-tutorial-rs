use ash::prelude::VkResult;
use ash::vk;

use crate::devices::Device;
use crate::swap_chain::SwapChain;

#[non_exhaustive]
pub struct SyncObjects {
    pub present_complete_semaphores: Vec<vk::Semaphore>,
    pub render_finished_semaphores: Vec<vk::Semaphore>,
    pub inflight_fences: Vec<vk::Fence>,
}

impl SyncObjects {
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
