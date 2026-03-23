use std::ffi::CStr;
use std::slice;

use anyhow::anyhow;
use ash::{khr, vk};

use crate::instance::Instance;
use crate::surface::Surface;

const REQUIRED_DEVICE_EXTENSIONS: &[&CStr] = &[khr::swapchain::NAME];

#[non_exhaustive]
pub struct PhysicalDevice {
    pub handle: vk::PhysicalDevice,
    pub queue_family: u32,
}

impl PhysicalDevice {
    pub fn new(instance: &Instance, surface: &Surface) -> anyhow::Result<Self> {
        let devices = unsafe { instance.handle.enumerate_physical_devices()? };
        devices
            .into_iter()
            .find_map(|device| {
                Self::check_device(&instance.handle, device, surface).map(|queue_family| Self {
                    handle: device,
                    queue_family,
                })
            })
            .ok_or_else(|| anyhow!("Couldn't find suitable physical device"))
    }

    /// Returns the queue family index if the device is suitable, `None` otherwise.
    fn check_device(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        surface: &Surface,
    ) -> Option<u32> {
        // Check Vulkan 1.3 support
        let properties = unsafe { instance.get_physical_device_properties(physical_device) };
        if properties.api_version < vk::API_VERSION_1_3 {
            return None;
        }

        // Check required extensions
        let available_extensions = unsafe {
            instance
                .enumerate_device_extension_properties(physical_device)
                .unwrap_or_default()
        };
        let supports_all_extensions = REQUIRED_DEVICE_EXTENSIONS.iter().all(|&required| {
            available_extensions
                .iter()
                .any(|available| available.extension_name_as_c_str().ok() == Some(required))
        });
        if !supports_all_extensions {
            return None;
        }

        // Check required features
        let mut vulkan_1_1_features = vk::PhysicalDeviceVulkan11Features::default();
        let mut vulkan_1_3_features = vk::PhysicalDeviceVulkan13Features::default();
        let mut extended_dynamic_state_features =
            vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT::default();
        let mut features = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut vulkan_1_1_features)
            .push_next(&mut vulkan_1_3_features)
            .push_next(&mut extended_dynamic_state_features);
        unsafe { instance.get_physical_device_features2(physical_device, &mut features) };
        let supports_all_features = vulkan_1_1_features.shader_draw_parameters == vk::TRUE
            && vulkan_1_3_features.dynamic_rendering == vk::TRUE
            && extended_dynamic_state_features.extended_dynamic_state == vk::TRUE;
        if !supports_all_features {
            return None;
        }

        // Find a queue family supporting both graphics and presentation
        let queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        queue_families
            .iter()
            .enumerate()
            .find(|&(qf, qfp)| {
                let supports_graphics = qfp.queue_flags.contains(vk::QueueFlags::GRAPHICS);
                let supports_presentation =
                    surface.is_queue_family_suitable(physical_device, qf as u32);
                supports_graphics && supports_presentation
            })
            .map(|(qf, _)| qf as u32)
    }

    pub fn name(&self, instance: &Instance) -> anyhow::Result<String> {
        let mut properties = vk::PhysicalDeviceProperties2::default();
        unsafe {
            instance
                .handle
                .get_physical_device_properties2(self.handle, &mut properties)
        };
        let name = properties
            .properties
            .device_name_as_c_str()?
            .to_string_lossy()
            .to_string();
        Ok(name)
    }
}

#[non_exhaustive]
pub struct Device {
    pub handle: ash::Device,
    pub queue: vk::Queue,
}

impl Device {
    pub fn new(instance: &Instance, physical_device: &PhysicalDevice) -> anyhow::Result<Self> {
        let queue_priorities = [0.5]; // implies queue_count == 1
        let queue_ci = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(physical_device.queue_family)
            .queue_priorities(&queue_priorities);

        let extension_ptrs: Vec<*const _> = REQUIRED_DEVICE_EXTENSIONS
            .iter()
            .map(|e| e.as_ptr())
            .collect();

        let mut vulkan_1_1_features =
            vk::PhysicalDeviceVulkan11Features::default().shader_draw_parameters(true);
        let mut vulkan_1_3_features =
            vk::PhysicalDeviceVulkan13Features::default().dynamic_rendering(true);
        let mut extended_dynamic_state_features =
            vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT::default()
                .extended_dynamic_state(true);
        let mut features = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut vulkan_1_1_features)
            .push_next(&mut vulkan_1_3_features)
            .push_next(&mut extended_dynamic_state_features);

        let device_ci = vk::DeviceCreateInfo::default()
            .queue_create_infos(slice::from_ref(&queue_ci))
            .enabled_extension_names(&extension_ptrs)
            .push_next(&mut features);

        let handle = unsafe {
            instance
                .handle
                .create_device(physical_device.handle, &device_ci, None)?
        };
        let queue = unsafe { handle.get_device_queue(physical_device.queue_family, 0) };

        Ok(Self { handle, queue })
    }

    /// # Safety
    ///
    /// - Must be called before the `ash::Instance` that was used to create this
    ///   device is destroyed.
    /// - Must be called after all child Vulkan objects (swapchains, pipelines, etc.)
    ///   have been destroyed.
    /// - Must be called at most once. Calling it more than once is undefined
    ///   behaviour as the underlying handle becomes invalid after the first call.
    pub unsafe fn destroy(&mut self) {
        unsafe { self.handle.destroy_device(None) };
    }
}
