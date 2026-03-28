use ash::vk;

use crate::devices::Device;

#[non_exhaustive]
pub struct SyncObjects {
    pub present_complete_semaphore: vk::Semaphore,
    pub render_finished_semaphore: vk::Semaphore,
    pub draw_fence: vk::Fence,
}

impl SyncObjects {
    pub fn new(device: &Device) -> anyhow::Result<Self> {
        let device_h = &device.handle;

        let semaphore_ci = vk::SemaphoreCreateInfo::default();
        let present_complete_semaphore = unsafe { device_h.create_semaphore(&semaphore_ci, None)? };
        let render_finished_semaphore = unsafe { device_h.create_semaphore(&semaphore_ci, None)? };

        let fence_ci = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        let draw_fence = unsafe { device_h.create_fence(&fence_ci, None)? };

        Ok(Self {
            present_complete_semaphore,
            render_finished_semaphore,
            draw_fence,
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
            device_h.destroy_semaphore(self.present_complete_semaphore, None);
            device_h.destroy_semaphore(self.render_finished_semaphore, None);
            device_h.destroy_fence(self.draw_fence, None);
        }
    }
}
