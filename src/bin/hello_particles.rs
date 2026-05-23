use std::slice;
use std::time::Instant;

use anyhow::bail;
use ash::vk;
use vulkan_tuto::buffers::{DeltaTime, StorageBuffers, UniformBuffers};
use vulkan_tuto::commands::ParticlesCommands;
use vulkan_tuto::descriptors::ParticlesDescriptors;
use vulkan_tuto::devices::{Device, PhysicalDevice};
use vulkan_tuto::instance::Instance;
use vulkan_tuto::pipelines::{ParticlesComputePipeline, ParticlesPipeline};
use vulkan_tuto::surface::Surface;
use vulkan_tuto::swap_chain::SwapChain;
use vulkan_tuto::sync::ParticlesSync;

struct HelloParticlesApp {
    last_frame_time: Instant,

    sdl_context: sdl3::Sdl,
    window: sdl3::video::Window,
    minimized: bool,
    frame_index: usize,

    instance: Instance,
    surface: Surface,
    physical_device: PhysicalDevice,
    device: Device,
    swap_chain: SwapChain,
    commands: ParticlesCommands,
    sync: ParticlesSync,

    descriptors: ParticlesDescriptors,
    uniform_buffers: UniformBuffers<DeltaTime>,
    storage_buffers: StorageBuffers,

    graphics_pipeline: ParticlesPipeline,
    compute_pipeline: ParticlesComputePipeline,
}

impl HelloParticlesApp {
    const WIDTH: u32 = 800;
    const HEIGHT: u32 = 600;
    const MAX_FRAMES_INFLIGHT: usize = 2;

    pub fn new() -> anyhow::Result<Self> {
        let sdl_context = sdl3::init()?;
        let video = sdl_context.video()?;
        let window = video
            .window("Hello Particles", Self::WIDTH, Self::HEIGHT)
            .vulkan()
            .resizable()
            .build()?;

        let instance = Instance::new(&window)?;
        let surface = Surface::new(&instance, &window)?;
        let physical_device = PhysicalDevice::new(&instance, &surface)?;
        let device = Device::new(&instance, &physical_device)?;
        let swap_chain = SwapChain::new(&instance, &physical_device, &device, &surface, &window)?;
        let commands =
            ParticlesCommands::new(&device, &physical_device, Self::MAX_FRAMES_INFLIGHT)?;
        let sync = ParticlesSync::new(&device, Self::MAX_FRAMES_INFLIGHT)?;

        let mut descriptors = ParticlesDescriptors::new(&device, Self::MAX_FRAMES_INFLIGHT)?;
        let storage_buffers = StorageBuffers::new(
            &instance,
            &physical_device,
            &device,
            &commands,
            Self::MAX_FRAMES_INFLIGHT,
            Self::WIDTH,
            Self::HEIGHT,
        )?;

        let uniform_buffers = UniformBuffers::new(
            &instance,
            &physical_device,
            &device,
            Self::MAX_FRAMES_INFLIGHT,
        )?;
        descriptors.allocate_desc_sets(&device, &uniform_buffers, &storage_buffers)?;

        let graphics_pipeline = ParticlesPipeline::new(
            &device,
            &swap_chain,
            concat!(env!("OUT_DIR"), "/particles_graphics.spv"),
        )?;

        let compute_pipeline = ParticlesComputePipeline::new(
            &device,
            &descriptors,
            concat!(env!("OUT_DIR"), "/particles_compute.spv"),
        )?;

        log::info!("Selected device: {physical_device:?}");

        Ok(Self {
            last_frame_time: Instant::now(),

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

            descriptors,
            uniform_buffers,
            storage_buffers,

            graphics_pipeline,
            compute_pipeline,
        })
    }

    fn draw_frame(&mut self) -> anyhow::Result<()> {
        let frame_start = Instant::now();

        let device_h = &self.device.handle;
        let inflight_fence = self.sync.inflight_fences[self.frame_index];

        // Acquire next swapchain image
        //
        // Signal the in-flight fence when the swapchain image is available,
        // so we know it is safe to re-record this frame slot's command buffers.
        let next_image = unsafe {
            self.swap_chain.fns.acquire_next_image(
                self.swap_chain.handle,
                u64::MAX,
                vk::Semaphore::null(),
                inflight_fence,
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

        // Wait for previous use of this frame slot
        unsafe {
            device_h.wait_for_fences(slice::from_ref(&inflight_fence), true, u64::MAX)?;
            device_h.reset_fences(slice::from_ref(&inflight_fence))?;
        }

        // compute_wait_value is read from self.sync.timeline_value which was last
        // written as graphics_signal_value of the previous frame. This is what
        // chains frames together: compute[N] cannot start until graphics[N-1]
        // has finished.
        let compute_wait_value = self.sync.timeline_value;
        let compute_signal_value = {
            self.sync.timeline_value += 1;
            self.sync.timeline_value
        };
        let graphics_wait_value = compute_signal_value;
        let graphics_signal_value = {
            self.sync.timeline_value += 1;
            self.sync.timeline_value
        };

        // Update UBO
        let delta_time = frame_start.duration_since(self.last_frame_time);
        self.uniform_buffers.update(self.frame_index, delta_time);

        // Submit compute — waits for N, signals N+1.
        // Ensures we don't overwrite particle positions while the previous
        // frame's graphics pass may still be reading them.
        self.commands.record_compute(
            &self.device,
            &self.compute_pipeline,
            &self.descriptors,
            self.frame_index,
        )?;

        let compute_wait_stage = vk::PipelineStageFlags::COMPUTE_SHADER;
        let mut compute_timeline_info = vk::TimelineSemaphoreSubmitInfo::default()
            .wait_semaphore_values(slice::from_ref(&compute_wait_value))
            .signal_semaphore_values(slice::from_ref(&compute_signal_value));

        let compute_submit_info = vk::SubmitInfo::default()
            .push_next(&mut compute_timeline_info)
            .wait_semaphores(slice::from_ref(&self.sync.semaphore))
            .wait_dst_stage_mask(slice::from_ref(&compute_wait_stage))
            .command_buffers(slice::from_ref(
                &self.commands.compute_cmd_buffers[self.frame_index],
            ))
            .signal_semaphores(slice::from_ref(&self.sync.semaphore));

        unsafe {
            device_h.queue_submit(
                self.device.queue,
                slice::from_ref(&compute_submit_info),
                vk::Fence::null(),
            )?;
        }

        // Submit graphics — waits for N+1, signals N+2.
        // Ensures we don't render particles before compute has finished
        // writing their new positions.
        self.commands.record_graphics(
            &self.device,
            &self.swap_chain,
            &self.graphics_pipeline,
            &self.storage_buffers,
            image_index as usize,
            self.frame_index,
        )?;

        let graphics_wait_stage = vk::PipelineStageFlags::VERTEX_INPUT;
        let mut graphics_timeline_info = vk::TimelineSemaphoreSubmitInfo::default()
            .wait_semaphore_values(slice::from_ref(&graphics_wait_value))
            .signal_semaphore_values(slice::from_ref(&graphics_signal_value));

        let graphics_submit_info = vk::SubmitInfo::default()
            .push_next(&mut graphics_timeline_info)
            .wait_semaphores(slice::from_ref(&self.sync.semaphore))
            .wait_dst_stage_mask(slice::from_ref(&graphics_wait_stage))
            .command_buffers(slice::from_ref(
                &self.commands.graphics_cmd_buffers[self.frame_index],
            ))
            .signal_semaphores(slice::from_ref(&self.sync.semaphore));

        unsafe {
            device_h.queue_submit(
                self.device.queue,
                slice::from_ref(&graphics_submit_info),
                vk::Fence::null(),
            )?;
        }

        // Block the CPU until graphics signals N+2, guaranteeing the
        // swapchain image is fully rendered before we hand it to the compositor.
        let semaphore_wait_info = vk::SemaphoreWaitInfo::default()
            .semaphores(slice::from_ref(&self.sync.semaphore))
            .values(slice::from_ref(&graphics_signal_value));
        unsafe {
            device_h.wait_semaphores(&semaphore_wait_info, u64::MAX)?;
        }
        // Present — no binary semaphore needed since we already waited on the
        // timeline above.
        let present_info = vk::PresentInfoKHR::default()
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

        self.last_frame_time = frame_start;
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
                self.last_frame_time = Instant::now();
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

impl Drop for HelloParticlesApp {
    fn drop(&mut self) {
        unsafe {
            self.compute_pipeline.destroy(&self.device);
            self.graphics_pipeline.destroy(&self.device);

            self.storage_buffers.destroy(&self.device);
            self.uniform_buffers.destroy(&self.device);
            self.descriptors.destroy(&self.device);

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
        env_logger::Env::default().filter_or("RUST_LOG", "vulkan=warn,vulkan_tuto=info,info"),
    );
    let app = HelloParticlesApp::new()?;
    app.run()?;
    Ok(())
}
