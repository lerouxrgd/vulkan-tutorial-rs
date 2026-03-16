use std::ffi::{CString, c_char};

use ash::vk;

const WIDTH: u32 = 1080;
const HEIGHT: u32 = 1080;

struct HelloTriangleApp {
    sdl_context: sdl3::Sdl,
    window: sdl3::video::Window,
    instance: ash::Instance,
}

impl HelloTriangleApp {
    pub fn new() -> anyhow::Result<Self> {
        // Initialize SDL
        let sdl_context = sdl3::init()?;

        // Load Vulkan library
        let entry = unsafe { ash::Entry::load()? };

        // Create window
        let video = sdl_context.video()?;
        let window: sdl3::video::Window = video
            .window("Vulkan tutorial", WIDTH, HEIGHT)
            .vulkan()
            .build()?;

        // App info
        let app_info = vk::ApplicationInfo::default()
            .application_name(c"Hello Triangle")
            .application_version(vk::make_api_version(0, 1, 0, 0))
            .engine_name(c"No Engine")
            .engine_version(vk::make_api_version(0, 1, 0, 0))
            .api_version(vk::API_VERSION_1_3);

        // Get the required instance extensions from SDL
        let extensions = window.vulkan_instance_extensions()?;
        let extension_names: Vec<CString> = extensions
            .into_iter()
            .map(|e| CString::new(e).unwrap_or_default())
            .collect();
        let extension_ptrs: Vec<*const c_char> =
            extension_names.iter().map(|e| e.as_ptr()).collect();

        // Instance
        let instance_ci = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&extension_ptrs);
        let instance = unsafe { entry.create_instance(&instance_ci, None)? };

        Ok(Self {
            sdl_context,
            window,
            instance,
        })
    }

    fn run(self) -> anyhow::Result<()> {
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
            self.instance.destroy_instance(None);
        }
    }
}

fn main() -> anyhow::Result<()> {
    let app = HelloTriangleApp::new()?;
    app.run()?;
    Ok(())
}
