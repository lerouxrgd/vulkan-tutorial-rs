use std::sync::mpsc;
use std::{slice, thread};

use ash::vk;
use bytemuck::{Pod, Zeroable};

use crate::descriptors::ParticlesDescriptors;
use crate::devices::{Device, PhysicalDevice};
use crate::pipelines::ParticlesComputeMtPipeline;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ParticleGroup {
    pub start_index: u32,
    pub count: u32,
}

pub struct WorkOrder {
    pub frame_index: usize,
    pub start_index: u32,
    pub count: u32,
}

pub struct WorkDone {
    pub cmd_buffer: vk::CommandBuffer,
}

pub struct WorkerThread {
    pub sender: mpsc::SyncSender<WorkOrder>,
    pub receiver: mpsc::Receiver<WorkDone>,
    pub handle: thread::JoinHandle<()>,
}

impl WorkerThread {
    pub fn spawn(
        device: &Device,
        physical_device: &PhysicalDevice,
        pipeline: &ParticlesComputeMtPipeline,
        descriptors: &ParticlesDescriptors,
        max_frames_inflight: usize,
    ) -> anyhow::Result<Self> {
        let device_h = device.handle.clone();

        let (order_tx, order_rx) = mpsc::sync_channel::<WorkOrder>(0);
        let (done_tx, done_rx) = mpsc::sync_channel::<WorkDone>(0);

        // Create per-thread command pool and buffer
        let pool_ci = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(physical_device.queue_family);
        let pool = unsafe { device_h.create_command_pool(&pool_ci, None)? };

        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(max_frames_inflight as u32);
        let cmd_buffers = unsafe { device_h.allocate_command_buffers(&alloc_info)? };

        let pipeline_h = pipeline.handle;
        let pipeline_layout = pipeline.layout;
        let desc_sets = descriptors.desc_sets.clone();

        let handle = thread::spawn(move || {
            while let Ok(order) = order_rx.recv() {
                let result = (|| -> anyhow::Result<()> {
                    let cmd_buffer = cmd_buffers[order.frame_index];
                    unsafe {
                        device_h.reset_command_buffer(
                            cmd_buffer,
                            vk::CommandBufferResetFlags::empty(),
                        )?;
                        device_h.begin_command_buffer(
                            cmd_buffer,
                            &vk::CommandBufferBeginInfo::default(),
                        )?;

                        device_h.cmd_bind_pipeline(
                            cmd_buffer,
                            vk::PipelineBindPoint::COMPUTE,
                            pipeline_h,
                        );

                        device_h.cmd_bind_descriptor_sets(
                            cmd_buffer,
                            vk::PipelineBindPoint::COMPUTE,
                            pipeline_layout,
                            0,
                            slice::from_ref(&desc_sets[order.frame_index]),
                            &[],
                        );

                        let push_constants = ParticleGroup {
                            start_index: order.start_index,
                            count: order.count,
                        };
                        device_h.cmd_push_constants(
                            cmd_buffer,
                            pipeline_layout,
                            vk::ShaderStageFlags::COMPUTE,
                            0,
                            bytemuck::bytes_of(&push_constants),
                        );

                        device_h.cmd_dispatch(cmd_buffer, order.count / 256, 1, 1);

                        device_h.end_command_buffer(cmd_buffer)?;
                    }

                    done_tx.send(WorkDone { cmd_buffer })?;
                    Ok(())
                })();

                if let Err(e) = result {
                    log::error!("Worker thread error: {e}");
                    break;
                }
            }

            unsafe {
                device_h.destroy_command_pool(pool, None);
            }
        });
        Ok(Self {
            sender: order_tx,
            receiver: done_rx,
            handle,
        })
    }
}
