mod buffers;
mod commands;
mod descriptors;
mod devices;
mod images;
mod instance;
mod logging;
mod pipeline;
mod surface;
mod swap_chain;
mod sync;

use std::slice;
use std::time::Instant;

use anyhow::bail;
use ash::vk;

use crate::buffers::{IndexBuffer, Model, UniformBuffers, VertexBuffer};
use crate::commands::Commands;
use crate::descriptors::Descriptors;
use crate::devices::{Device, PhysicalDevice};
use crate::images::{TextureImage, TextureSampler};
use crate::instance::Instance;
use crate::pipeline::GraphicsPipeline;
use crate::surface::Surface;
use crate::swap_chain::SwapChain;
use crate::sync::SyncObjects;

struct HelloTriangleApp {
    start_time: Instant,

    sdl_context: sdl3::Sdl,
    window: sdl3::video::Window,
    minimized: bool,
    frame_index: usize,

    instance: Instance,
    surface: Surface,
    physical_device: PhysicalDevice,
    device: Device,
    swap_chain: SwapChain,
    commands: Commands,
    sync: SyncObjects,

    vertex_buffer: VertexBuffer,
    index_buffer: IndexBuffer,

    descriptors: Descriptors,
    uniform_buffers: UniformBuffers,
    texture_image: TextureImage,
    texture_sampler: TextureSampler,

    pipeline: GraphicsPipeline,
}

impl HelloTriangleApp {
    const WIDTH: u32 = 800;
    const HEIGHT: u32 = 600;
    const MAX_FRAMES_INFLIGHT: usize = 2;

    pub fn new() -> anyhow::Result<Self> {
        let sdl_context = sdl3::init()?;
        let video = sdl_context.video()?;
        let window = video
            .window("Hello Triangle", Self::WIDTH, Self::HEIGHT)
            .vulkan()
            .resizable()
            .build()?;

        let instance = Instance::new(&window)?;
        let surface = Surface::new(&instance, &window)?;
        let physical_device = PhysicalDevice::new(&instance, &surface)?;
        let device = Device::new(&instance, &physical_device)?;
        let swap_chain = SwapChain::new(&instance, &physical_device, &device, &surface, &window)?;
        let commands = Commands::new(&device, &physical_device, Self::MAX_FRAMES_INFLIGHT)?;
        let sync = SyncObjects::new(&device, &swap_chain, Self::MAX_FRAMES_INFLIGHT)?;

        let model = Model::from_obj("assets/models/viking_room.obj")?;
        let vertex_buffer = VertexBuffer::new(
            &instance,
            &physical_device,
            &device,
            &commands,
            &model.vertices,
        )?;
        let index_buffer = IndexBuffer::new(
            &instance,
            &physical_device,
            &device,
            &commands,
            &model.indices,
        )?;

        let mut descriptors = Descriptors::new(&device, Self::MAX_FRAMES_INFLIGHT)?;
        let uniform_buffers = UniformBuffers::new(
            &instance,
            &physical_device,
            &device,
            Self::MAX_FRAMES_INFLIGHT,
        )?;
        let texture_image = TextureImage::new(
            &instance,
            &physical_device,
            &device,
            &commands,
            "assets/textures/viking_room.png",
        )?;
        let texture_sampler = TextureSampler::new(&instance, &physical_device, &device)?;
        descriptors.allocate_desc_sets(
            &device,
            &uniform_buffers,
            &texture_image,
            &texture_sampler,
        )?;

        let pipeline = GraphicsPipeline::new(
            &device,
            &swap_chain,
            &descriptors,
            concat!(env!("OUT_DIR"), "/slang.spv"),
        )?;

        log::info!("Selected device: {}", physical_device.name(&instance)?);

        Ok(Self {
            start_time: Instant::now(),

            sdl_context,
            window,
            minimized: false,
            frame_index: 0,

            instance,
            surface,
            physical_device,
            device,
            swap_chain,
            commands,
            sync,

            vertex_buffer,
            index_buffer,

            descriptors,
            uniform_buffers,
            texture_image,
            texture_sampler,

            pipeline,
        })
    }

    fn draw_frame(&mut self) -> anyhow::Result<()> {
        let time_delta = self.start_time.elapsed().as_secs_f32();

        let device_h = &self.device.handle;
        let inflight_fence = self.sync.inflight_fences[self.frame_index];
        let present_complete_semaphore = self.sync.present_complete_semaphores[self.frame_index];
        let command_buffer = self.commands.buffers[self.frame_index];

        // Wait for the previous frame to finish
        unsafe { device_h.wait_for_fences(slice::from_ref(&inflight_fence), true, u64::MAX)? }

        // Acquire next swapchain image
        let next_image = unsafe {
            self.swap_chain.fns.acquire_next_image(
                self.swap_chain.handle,
                u64::MAX,
                present_complete_semaphore,
                vk::Fence::null(),
            )
        };
        let image_index = match next_image {
            Ok((image_index, suboptimal)) => {
                if suboptimal {
                    log::warn!("acquire_next_image returned suboptimal");
                }
                image_index
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                unsafe {
                    self.swap_chain.recreate(
                        &self.instance,
                        &self.physical_device,
                        &self.device,
                        &self.surface,
                        &self.window,
                    )?
                };
                return Ok(());
            }
            Err(e) => bail!("Failed to acquire acquire_next_image: {e}"),
        };
        let render_finished_semaphore = self.sync.render_finished_semaphores[image_index as usize];

        self.uniform_buffers
            .update(self.frame_index, time_delta, &self.swap_chain);

        // Only reset the fence if we are submitting work
        unsafe { device_h.reset_fences(slice::from_ref(&inflight_fence))? }

        // Record commands for this frame
        self.commands.record(
            &self.device,
            &self.swap_chain,
            &self.pipeline,
            &self.vertex_buffer,
            &self.index_buffer,
            &self.descriptors,
            image_index as usize,
            self.frame_index,
        )?;

        // Submit command buffer
        let wait_dst_stage_mask = vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT; // will be gated by semaphore
        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(slice::from_ref(&present_complete_semaphore))
            .wait_dst_stage_mask(slice::from_ref(&wait_dst_stage_mask))
            .command_buffers(slice::from_ref(&command_buffer))
            .signal_semaphores(slice::from_ref(&render_finished_semaphore));
        unsafe {
            device_h.queue_submit(
                self.device.queue,
                slice::from_ref(&submit_info),
                inflight_fence,
            )?
        };

        // Present image
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(slice::from_ref(&render_finished_semaphore))
            .swapchains(slice::from_ref(&self.swap_chain.handle))
            .image_indices(slice::from_ref(&image_index));
        let presentation = unsafe {
            self.swap_chain
                .fns
                .queue_present(self.device.queue, &present_info)
        };
        match presentation {
            Ok(false) => {}
            Ok(true) => log::warn!("queue_present returned suboptimal"),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                unsafe {
                    self.swap_chain.recreate(
                        &self.instance,
                        &self.physical_device,
                        &self.device,
                        &self.surface,
                        &self.window,
                    )?
                };
            }
            Err(e) => bail!(e),
        }

        self.frame_index = (self.frame_index + 1) % Self::MAX_FRAMES_INFLIGHT;
        Ok(())
    }

    fn handle_event(&mut self, event: sdl3::event::Event) -> anyhow::Result<bool> {
        match event {
            // Quit
            sdl3::event::Event::Quit { .. } => {
                return Ok(true);
            }

            // Handle window minimization/restoration
            sdl3::event::Event::Window {
                win_event: sdl3::event::WindowEvent::Minimized | sdl3::event::WindowEvent::Occluded,
                ..
            } => {
                log::debug!("Window minimized/occluded");
                self.minimized = true;
            }
            sdl3::event::Event::Window {
                win_event: sdl3::event::WindowEvent::Restored | sdl3::event::WindowEvent::Exposed,
                ..
            } => {
                log::debug!("Window restored/exposed");
                self.minimized = false;
            }

            // Handle window resizing
            sdl3::event::Event::Window {
                win_event: sdl3::event::WindowEvent::Resized(..),
                ..
            } => {
                unsafe {
                    self.swap_chain.recreate(
                        &self.instance,
                        &self.physical_device,
                        &self.device,
                        &self.surface,
                        &self.window,
                    )?
                };
            }

            _ => {}
        }

        Ok(false)
    }

    pub fn run(mut self) -> anyhow::Result<()> {
        let mut event_pump = self.sdl_context.event_pump()?;

        'running: loop {
            // When minimized, block until an event arrives
            let first_event = if self.minimized {
                Some(event_pump.wait_event())
            } else {
                None
            };

            // Process SDL3 events
            for event in first_event.into_iter().chain(event_pump.poll_iter()) {
                let done = self.handle_event(event)?;
                if done {
                    break 'running;
                }
            }

            // Draw frame if the app is not minimized
            if !self.minimized {
                self.draw_frame()?;
            }
        }

        // Finish device operations before destroying resources (through Drop impl)
        unsafe { self.device.handle.device_wait_idle()? };

        Ok(())
    }
}

impl Drop for HelloTriangleApp {
    fn drop(&mut self) {
        unsafe {
            self.pipeline.destroy(&self.device);

            self.texture_sampler.destroy(&self.device);
            self.texture_image.destroy(&self.device);
            self.uniform_buffers.destroy(&self.device);
            self.descriptors.destroy(&self.device);

            self.index_buffer.destroy(&self.device);
            self.vertex_buffer.destroy(&self.device);

            self.sync.destroy(&self.device);
            self.commands.destroy(&self.device);
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
