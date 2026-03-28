mod commands;
mod devices;
mod instance;
mod logging;
mod pipeline;
mod surface;
mod swap_chain;
mod sync;

use std::slice;

use anyhow::bail;
use ash::vk;

use crate::commands::Commands;
use crate::devices::{Device, PhysicalDevice};
use crate::instance::Instance;
use crate::pipeline::GraphicsPipeline;
use crate::surface::Surface;
use crate::swap_chain::SwapChain;
use crate::sync::SyncObjects;

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
    sync: SyncObjects,
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
        let sync = SyncObjects::new(&device)?;

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
            sync,
        })
    }

    fn draw(&mut self) -> anyhow::Result<()> {
        let device_h = &self.device.handle;

        // Wait for the previous frame to finish
        unsafe {
            device_h.wait_for_fences(slice::from_ref(&self.sync.draw_fence), true, u64::MAX)?;
            device_h.reset_fences(slice::from_ref(&self.sync.draw_fence))?;
        }

        // Acquire next swapchain image
        let (image_index, _suboptimal) = unsafe {
            self.swap_chain.fns.acquire_next_image(
                self.swap_chain.handle,
                u64::MAX,
                self.sync.present_complete_semaphore,
                vk::Fence::null(),
            )?
        };

        // Record commands for this frame
        self.commands.record(
            &self.device,
            &self.swap_chain,
            &self.pipeline,
            image_index as usize,
        )?;

        // NOTE: for simplicity, wait for the queue to be idle before submitting
        unsafe { device_h.queue_wait_idle(self.device.queue)? };

        // Submit command buffer
        let wait_dst_stage_mask = vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT; // will be gated by semaphore
        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(slice::from_ref(&self.sync.present_complete_semaphore))
            .wait_dst_stage_mask(slice::from_ref(&wait_dst_stage_mask))
            .command_buffers(slice::from_ref(&self.commands.buffer))
            .signal_semaphores(slice::from_ref(&self.sync.render_finished_semaphore));
        unsafe {
            device_h.queue_submit(
                self.device.queue,
                slice::from_ref(&submit_info),
                self.sync.draw_fence,
            )?
        };

        // Present
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(slice::from_ref(&self.sync.render_finished_semaphore))
            .swapchains(slice::from_ref(&self.swap_chain.handle))
            .image_indices(slice::from_ref(&image_index));
        let result = unsafe {
            self.swap_chain
                .fns
                .queue_present(self.device.queue, &present_info)
        };
        match result {
            Ok(false) => {}
            Ok(true) => log::warn!("queue_present returned suboptimal"),
            Err(e) => bail!("queue_present failed: {e}"),
        }

        Ok(())
    }

    pub fn run(mut self) -> anyhow::Result<()> {
        let mut event_pump = self.sdl_context.event_pump()?;
        let mut quit = false;
        while !quit {
            self.draw()?;
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
        unsafe { self.device.handle.device_wait_idle()? }; // finish device operations before destroying resources
        Ok(())
    }
}

impl Drop for HelloTriangleApp {
    fn drop(&mut self) {
        unsafe {
            self.sync.destroy(&self.device);
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
