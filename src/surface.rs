use std::ptr;

use anyhow::ensure;
use ash::{khr, vk};
use sdl3::sys::vulkan::SDL_Vulkan_CreateSurface;

use crate::instance::Instance;

#[non_exhaustive]
pub struct Surface {
    pub fns: khr::surface::Instance,
    pub handle: vk::SurfaceKHR,
}

impl Surface {
    pub fn new(instance: &Instance, window: &sdl3::video::Window) -> anyhow::Result<Self> {
        let mut handle = vk::SurfaceKHR::null();
        unsafe {
            ensure!(
                SDL_Vulkan_CreateSurface(
                    window.raw(),
                    instance.raw_handle(),
                    ptr::null(),
                    &mut handle
                ),
                "SDL_Vulkan_CreateSurface failed"
            );
        }
        let fns = khr::surface::Instance::new(&instance.entry, &instance.handle);
        Ok(Self { fns, handle })
    }

    pub fn is_queue_family_suitable(
        &self,
        physical_device: vk::PhysicalDevice,
        queue_family: u32,
    ) -> bool {
        unsafe {
            self.fns
                .get_physical_device_surface_support(physical_device, queue_family, self.handle)
                .inspect_err(|&e| {
                    log::warn!(
                        "Couldn't get surface support for device={physical_device:?} qf={queue_family}: {e}"
                    );

                })
                .unwrap_or(false)
        }
    }

    /// # Safety
    ///
    /// - Must be called before the `ash::Instance` that was used to create this
    ///   `Surface` is destroyed.
    /// - Must be called at most once. Calling it more than once is undefined
    ///   behaviour as the underlying handle becomes invalid after the first call.
    pub unsafe fn destroy(&mut self) {
        unsafe { self.fns.destroy_surface(self.handle, None) };
    }
}
