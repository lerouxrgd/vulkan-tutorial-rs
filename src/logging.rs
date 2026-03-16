use std::ffi::{CStr, c_void};

use ash::ext::debug_utils;
use ash::vk;

pub struct DebugUtils {
    loader: debug_utils::Instance,
    messenger: vk::DebugUtilsMessengerEXT,
}

impl DebugUtils {
    pub fn new(entry: &ash::Entry, instance: &ash::Instance) -> anyhow::Result<Self> {
        let loader = debug_utils::Instance::new(entry, instance);
        let ci = DebugUtils::messenger_ci();
        let messenger = unsafe { loader.create_debug_utils_messenger(&ci, None)? };
        Ok(Self { loader, messenger })
    }

    /// Builds a `[vk::DebugUtilsMessengerCreateInfoEXT]()` wired to a Rust logging callback.
    ///
    /// Returned by value so it can be used both for the instance-creation chain and for the
    /// persistent messenger.
    pub fn messenger_ci() -> vk::DebugUtilsMessengerCreateInfoEXT<'static> {
        vk::DebugUtilsMessengerCreateInfoEXT::default()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                    | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                    | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                    | vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
            )
            .pfn_user_callback(Some(vulkan_debug_callback))
    }

    /// # Safety
    ///
    /// - Must be called before the `ash::Instance` that was used to create this
    ///   `DebugUtils` is destroyed.
    /// - Must be called at most once. Calling it more than once is undefined
    ///   behaviour as the underlying handle becomes invalid after the first call.
    pub unsafe fn destroy(&mut self) {
        unsafe {
            self.loader
                .destroy_debug_utils_messenger(self.messenger, None)
        };
    }
}

/// The actual Vulkan debug callback. Severity is mapped to the matching `log` level.
unsafe extern "system" fn vulkan_debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    msg_type: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut c_void,
) -> vk::Bool32 {
    let message = unsafe {
        CStr::from_ptr((*data).p_message)
            .to_str()
            .unwrap_or("<invalid UTF-8>")
    };

    let type_tag = match msg_type {
        vk::DebugUtilsMessageTypeFlagsEXT::GENERAL => "general",
        vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION => "validation",
        vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE => "performance",
        _ => "unknown",
    };

    match severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => {
            log::error!(target: "vulkan", "[{type_tag}] {message}");
        }
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => {
            log::warn!(target: "vulkan", "[{type_tag}] {message}");
        }
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => {
            log::info!(target: "vulkan", "[{type_tag}] {message}");
        }
        _ => {
            // VERBOSE
            log::debug!(target: "vulkan", "[{type_tag}] {message}");
        }
    }

    vk::FALSE // Don't abort the Vulkan call that triggered this
}
