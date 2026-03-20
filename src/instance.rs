use std::ffi::{CString, c_char};

use ash::ext::debug_utils;
use ash::vk;

use crate::logging::DebugUtils;

#[non_exhaustive]
pub struct Instance {
    pub entry: ash::Entry,
    pub handle: ash::Instance,
    pub debug_utils: Option<DebugUtils>,
}

impl Instance {
    pub fn new(window: &sdl3::video::Window) -> anyhow::Result<Self> {
        let entry = unsafe { ash::Entry::load()? };

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
        let handle = unsafe { entry.create_instance(&instance_ci, None)? };

        let debug_utils = if cfg!(debug_assertions) {
            Some(DebugUtils::new(&entry, &handle)?)
        } else {
            None
        };

        Ok(Self {
            entry,
            handle,
            debug_utils,
        })
    }

    pub fn raw_handle(&self) -> vk::Instance {
        self.handle.handle()
    }

    /// # Safety
    ///
    /// - Must be called after all child Vulkan objects (devices, surfaces, etc.)
    ///   have been destroyed.
    /// - Must be called at most once. Calling it more than once is undefined
    ///   behaviour as the underlying handle becomes invalid after the first call.
    pub unsafe fn destroy(&mut self) {
        if let Some(mut debug_utils) = self.debug_utils.take() {
            unsafe { debug_utils.destroy() };
        }
        unsafe { self.handle.destroy_instance(None) };
    }
}
