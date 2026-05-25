use std::slice;
use std::time::Instant;

use anyhow::bail;
use ash::vk;
use vulkan_tuto::buffers::{DeltaTime, StorageBuffers, UniformBuffers};
use vulkan_tuto::commands::ParticlesMtCommands;
use vulkan_tuto::descriptors::ParticlesDescriptors;
use vulkan_tuto::devices::{Device, PhysicalDevice};
use vulkan_tuto::instance::Instance;
use vulkan_tuto::pipelines::{ParticlesComputeMtPipeline, ParticlesPipeline};
use vulkan_tuto::surface::Surface;
use vulkan_tuto::swap_chain::SwapChain;
use vulkan_tuto::sync::ParticlesMtSync;
use vulkan_tuto::worker::{WorkOrder, WorkerThread};

const THREAD_COUNT: usize = 8;
const PARTICLE_COUNT: usize = 8192;
const PARTICLES_PER_THREAD: u32 = (PARTICLE_COUNT / THREAD_COUNT) as u32;

struct HelloParticlesMtApp {
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
    commands: ParticlesMtCommands,
    sync: ParticlesMtSync,

    descriptors: ParticlesDescriptors,
    uniform_buffers: UniformBuffers<DeltaTime>,
    storage_buffers: StorageBuffers,

    graphics_pipeline: ParticlesPipeline,
    compute_pipeline: ParticlesComputeMtPipeline,

    workers: Vec<WorkerThread>,
}

impl HelloParticlesMtApp {
    const WIDTH: u32 = 800;
    const HEIGHT: u32 = 600;
    const MAX_FRAMES_INFLIGHT: usize = 2;

    pub fn new() -> anyhow::Result<Self> {
        let sdl_context = sdl3::init()?;
        let video = sdl_context.video()?;
        let window = video
            .window("Hello Particles MT", Self::WIDTH, Self::HEIGHT)
            .vulkan()
            .resizable()
            .build()?;

        let instance = Instance::new(&window)?;
        let surface = Surface::new(&instance, &window)?;
        let physical_device = PhysicalDevice::new(&instance, &surface)?;
        let device = Device::new(&instance, &physical_device)?;
        let swap_chain = SwapChain::new(&instance, &physical_device, &device, &surface, &window)?;
        let commands =
            ParticlesMtCommands::new(&device, &physical_device, Self::MAX_FRAMES_INFLIGHT)?;
        let sync = ParticlesMtSync::new(&device, &swap_chain, Self::MAX_FRAMES_INFLIGHT)?;

        let storage_buffers = StorageBuffers::new(
            &instance,
            &physical_device,
            &device,
            commands.pool,
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

        let mut descriptors = ParticlesDescriptors::new(&device, Self::MAX_FRAMES_INFLIGHT)?;
        descriptors.allocate_desc_sets(&device, &uniform_buffers, &storage_buffers)?;

        let graphics_pipeline = ParticlesPipeline::new(
            &device,
            &swap_chain,
            concat!(env!("OUT_DIR"), "/particles_multithread_graphics.spv"),
        )?;

        let compute_pipeline = ParticlesComputeMtPipeline::new(
            &device,
            &descriptors,
            concat!(env!("OUT_DIR"), "/particles_multithread_compute.spv"),
        )?;

        // Spawn worker threads
        let workers = (0..THREAD_COUNT)
            .map(|_| {
                WorkerThread::spawn(
                    &device,
                    &physical_device,
                    &compute_pipeline,
                    &descriptors,
                    Self::MAX_FRAMES_INFLIGHT,
                )
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

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
            workers,
        })
    }

    fn draw_frame(&mut self) -> anyhow::Result<()> {
        let frame_start = Instant::now();
        let device_h = &self.device.handle;
        let frame_index = self.frame_index;

        let inflight_fence = self.sync.inflight_fences[frame_index];
        let image_available = self.sync.image_available_semaphores[frame_index];
        let compute_finished = self.sync.compute_finished_semaphores[frame_index];

        // Wait for previous frame to finish
        unsafe {
            device_h.wait_for_fences(slice::from_ref(&inflight_fence), true, u64::MAX)?;
        }

        // Acquire next swapchain image
        let next_image = unsafe {
            self.swap_chain.fns.acquire_next_image(
                self.swap_chain.handle,
                u64::MAX,
                image_available,
                vk::Fence::null(),
            )
        };

        let image_index = match next_image {
            Ok((index, suboptimal)) => {
                if suboptimal {
                    log::warn!("acquire_next_image: suboptimal");
                }
                index
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
            Err(e) => bail!("acquire_next_image failed: {e}"),
        };

        unsafe {
            device_h.reset_fences(slice::from_ref(&inflight_fence))?;
        }

        // Update UBO
        let delta = frame_start.duration_since(self.last_frame_time);
        self.uniform_buffers.update(frame_index, delta);

        // Signal all workers to start recording compute command buffers
        for (i, worker) in self.workers.iter().enumerate() {
            let start_index = i as u32 * PARTICLES_PER_THREAD;
            let count = if i == THREAD_COUNT - 1 {
                PARTICLE_COUNT as u32 - start_index
            } else {
                PARTICLES_PER_THREAD
            };
            worker.sender.send(WorkOrder {
                frame_index,
                start_index,
                count,
            })?;
        }

        // Record graphics command buffer on main thread in parallel with workers
        self.commands.record_graphics(
            &self.device,
            &self.swap_chain,
            &self.graphics_pipeline,
            &self.storage_buffers,
            image_index as usize,
            frame_index,
        )?;

        // Collect compute command buffers from all workers
        let compute_cmd_buffers = self
            .workers
            .iter()
            .map(|w| w.receiver.recv().map(|done| done.cmd_buffer))
            .collect::<Result<Vec<_>, _>>()?;

        // Submit compute work — signals compute_finished
        let compute_submit_info = vk::SubmitInfo::default()
            .command_buffers(&compute_cmd_buffers)
            .signal_semaphores(slice::from_ref(&compute_finished));

        unsafe {
            device_h.queue_submit(
                self.device.queue,
                slice::from_ref(&compute_submit_info),
                vk::Fence::null(),
            )?;
        }

        // Submit graphics — waits on compute_finished + image_available,
        // signals render_finished, signals fence
        let render_finished = self.sync.render_finished_semaphores[image_index as usize];
        let graphics_wait_semaphores = [compute_finished, image_available];
        let graphics_wait_stages = [
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        ];
        let graphics_cmd_buffer = self.commands.graphics_cmd_buffers[frame_index];

        let graphics_submit_info = vk::SubmitInfo::default()
            .wait_semaphores(&graphics_wait_semaphores)
            .wait_dst_stage_mask(&graphics_wait_stages)
            .command_buffers(slice::from_ref(&graphics_cmd_buffer))
            .signal_semaphores(slice::from_ref(&render_finished));

        unsafe {
            device_h.queue_submit(
                self.device.queue,
                slice::from_ref(&graphics_submit_info),
                inflight_fence,
            )?;
        }

        // Present
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(slice::from_ref(&render_finished))
            .swapchains(slice::from_ref(&self.swap_chain.handle))
            .image_indices(slice::from_ref(&image_index));

        let present_result = unsafe {
            self.swap_chain
                .fns
                .queue_present(self.device.queue, &present_info)
        };

        match present_result {
            Ok(false) => {}
            Ok(true) => log::warn!("queue_present: suboptimal"),
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
            sdl3::event::Event::Quit { .. } => return Ok(true),
            sdl3::event::Event::Window {
                win_event: sdl3::event::WindowEvent::Minimized | sdl3::event::WindowEvent::Occluded,
                ..
            } => {
                log::debug!("Window minimized/occluded");
                unsafe { self.device.handle.device_wait_idle().ok() };
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
            let first_event = if self.minimized {
                Some(event_pump.wait_event())
            } else {
                None
            };

            for event in first_event.into_iter().chain(event_pump.poll_iter()) {
                let done = self.handle_event(event)?;
                if done {
                    break 'running;
                }
            }

            if !self.minimized {
                self.draw_frame()?;
            }
        }

        unsafe { self.device.handle.device_wait_idle()? };
        Ok(())
    }
}

impl Drop for HelloParticlesMtApp {
    fn drop(&mut self) {
        unsafe {
            self.device.handle.device_wait_idle().ok();

            // Drop workers first — closing sender channels causes worker threads to exit
            self.workers.clear();

            self.compute_pipeline.destroy(&self.device);
            self.graphics_pipeline.destroy(&self.device);
            self.descriptors.destroy(&self.device);
            self.uniform_buffers.destroy(&self.device);
            self.storage_buffers.destroy(&self.device);
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
    env_logger::init();
    let app = HelloParticlesMtApp::new()?;
    app.run()?;
    Ok(())
}
