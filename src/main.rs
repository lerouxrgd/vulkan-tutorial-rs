mod commands;
mod devices;
mod instance;
mod logging;
mod pipeline;
mod surface;
mod swap_chain;

use crate::commands::Commands;
use crate::devices::{Device, PhysicalDevice};
use crate::instance::Instance;
use crate::pipeline::GraphicsPipeline;
use crate::surface::Surface;
use crate::swap_chain::SwapChain;

struct HelloTriangleApp {
    sdl_context: sdl3::Sdl,
    window: sdl3::video::Window,
    instance: Instance,
    surface: Surface,
    physical_device: PhysicalDevice,
    device: Device,
    swap_chain: SwapChain,
    pipeline: GraphicsPipeline,
    commands: Commands,
}

impl HelloTriangleApp {
    const WIDTH: u32 = 800;
    const HEIGHT: u32 = 600;

    pub fn new() -> anyhow::Result<Self> {
        let sdl_context = sdl3::init()?;
        let video = sdl_context.video()?;
        let window = video
            .window("Hello Triangle", Self::WIDTH, Self::HEIGHT)
            .vulkan()
            .build()?;

        let instance = Instance::new(&window)?;
        let surface = Surface::new(&instance, &window)?;
        let physical_device = PhysicalDevice::new(&instance, &surface)?;
        let device = Device::new(&instance, &physical_device)?;
        let swap_chain = SwapChain::new(&instance, &physical_device, &device, &surface, &window)?;
        let pipeline =
            GraphicsPipeline::new(&device, &swap_chain, concat!(env!("OUT_DIR"), "/slang.spv"))?;
        let commands = Commands::new(&device, &physical_device)?;

        log::info!("Selected device: {}", physical_device.name(&instance)?);

        Ok(Self {
            sdl_context,
            window,
            instance,
            surface,
            physical_device,
            device,
            swap_chain,
            pipeline,
            commands,
        })
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
            self.commands.destroy(&self.device);
            self.pipeline.destroy(&self.device);
            self.swap_chain.destroy(&self.device);
            self.device.destroy();
            self.surface.destroy();
            self.instance.destroy();
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
