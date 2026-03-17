mod logging;

use std::ffi::{CStr, CString, c_char};

use anyhow::anyhow;
use ash::ext::debug_utils;
use ash::{khr, vk};

use crate::logging::DebugUtils;

const WIDTH: u32 = 1080;
const HEIGHT: u32 = 1080;

struct HelloTriangleApp {
    sdl_context: sdl3::Sdl,
    window: sdl3::video::Window,
    entry: ash::Entry,
    instance: ash::Instance,
    debug_utils: Option<DebugUtils>,
    physical_device: vk::PhysicalDevice,
}

impl HelloTriangleApp {
    pub fn new() -> anyhow::Result<Self> {
        // Initialize SDL
        let sdl_context = sdl3::init()?;

        // Create window
        let video = sdl_context.video()?;
        let window: sdl3::video::Window = video
            .window("Hello Triangle", WIDTH, HEIGHT)
            .vulkan()
            .build()?;

        // Load Vulkan library
        let entry = unsafe { ash::Entry::load()? };

        // Create instance
        let app_info = vk::ApplicationInfo::default()
            .application_name(c"Hello Triangle")
            .application_version(vk::make_api_version(0, 1, 0, 0))
            .engine_name(c"No Engine")
            .engine_version(vk::make_api_version(0, 1, 0, 0))
            .api_version(vk::API_VERSION_1_3);
        let mut extensions = window.vulkan_instance_extensions()?; // Required by SDL
        if cfg!(debug_assertions) {
            extensions.push(debug_utils::NAME.to_string_lossy().into());
        }
        let extension_names: Vec<CString> = extensions
            .into_iter()
            .map(|e| CString::new(e).unwrap_or_default())
            .collect();
        let extension_ptrs: Vec<*const c_char> =
            extension_names.iter().map(|e| e.as_ptr()).collect();
        let layer_names = [c"VK_LAYER_KHRONOS_validation".as_ptr()];
        let mut debug_ci = DebugUtils::messenger_ci();
        let mut instance_ci = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&extension_ptrs);
        if cfg!(debug_assertions) {
            instance_ci = instance_ci
                .enabled_layer_names(&layer_names)
                .push_next(&mut debug_ci);
        }
        let instance = unsafe { entry.create_instance(&instance_ci, None)? };

        // Set up optional debugger
        let debug_utils = if cfg!(debug_assertions) {
            Some(DebugUtils::new(&entry, &instance)?)
        } else {
            None
        };

        // Setup device
        let devices = unsafe { instance.enumerate_physical_devices()? };
        let physical_device = devices
            .into_iter()
            .find(|&device| Self::is_device_suitable(&instance, device))
            .ok_or_else(|| anyhow!("Couldn't find suitable physical device"))?;
        let mut device_properties = vk::PhysicalDeviceProperties2::default();
        unsafe {
            instance.get_physical_device_properties2(physical_device, &mut device_properties)
        };
        let device_name = device_properties
            .properties
            .device_name_as_c_str()?
            .to_string_lossy();
        log::info!("Selected device: {device_name}");

        Ok(Self {
            sdl_context,
            window,
            entry,
            instance,
            debug_utils,
            physical_device,
        })
    }

    fn is_device_suitable(instance: &ash::Instance, physical_device: vk::PhysicalDevice) -> bool {
        // Check if the physical device supports the Vulkan 1.3 API version
        let properties = unsafe { instance.get_physical_device_properties(physical_device) };
        let supports_vulkan_1_3 = properties.api_version >= vk::API_VERSION_1_3;

        // Check if any of the queue families support graphics operations
        let queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        let supports_graphics = queue_families
            .iter()
            .any(|qfp| qfp.queue_flags.contains(vk::QueueFlags::GRAPHICS));

        // Check if all required device extensions are available
        const REQUIRED_DEVICE_EXTENSIONS: &[&CStr] = &[khr::swapchain::NAME];
        let available_extensions = unsafe {
            instance
                .enumerate_device_extension_properties(physical_device)
                .unwrap_or_default()
        };
        let supports_all_required_extensions = REQUIRED_DEVICE_EXTENSIONS.iter().all(|&required| {
            available_extensions
                .iter()
                .any(|available| available.extension_name_as_c_str().ok() == Some(required))
        });

        // Check if the physical device supports the required features
        let mut vulkan_1_3_features = vk::PhysicalDeviceVulkan13Features::default();
        let mut extended_dynamic_state_features =
            vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT::default();
        let mut features = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut vulkan_1_3_features)
            .push_next(&mut extended_dynamic_state_features);
        unsafe { instance.get_physical_device_features2(physical_device, &mut features) };
        let supports_required_features = vulkan_1_3_features.dynamic_rendering == vk::TRUE
            && extended_dynamic_state_features.extended_dynamic_state == vk::TRUE;

        supports_vulkan_1_3
            && supports_graphics
            && supports_all_required_extensions
            && supports_required_features
    }

    pub fn run(self) -> anyhow::Result<()> {
        let mut event_pump = self.sdl_context.event_pump()?;
        let mut quit = false;
        while !quit {
            for event in event_pump.poll_iter() {
                match event {
                    sdl3::event::Event::Quit { .. } => {
                        quit = true;
                        break;
                    }
                    _ => {}
                }
            }
        }

        Ok(())
    }
}

impl Drop for HelloTriangleApp {
    fn drop(&mut self) {
        unsafe {
            if let Some(mut debug_utils) = self.debug_utils.take() {
                debug_utils.destroy();
            }
            self.instance.destroy_instance(None);
        }
    }
}

fn main() -> anyhow::Result<()> {
    env_logger::init_from_env(
        env_logger::Env::default().filter_or("RUST_LOG", "vulkan=warn,vulkan_tuto_rs=info,info"),
    );
    let app = HelloTriangleApp::new()?;
    app.run()?;
    Ok(())
}
