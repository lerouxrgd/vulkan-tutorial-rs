use std::slice;

use ash::prelude::VkResult;
use ash::vk;

use crate::buffers::{IndexBuffer, Particle, StorageBuffers, VertexBuffer};
use crate::descriptors::ParticlesDescriptors;
use crate::devices::{Device, PhysicalDevice};
use crate::game_object::GameObject;
use crate::images::transition_image_layout;
use crate::pipelines::{ParticlesComputePipeline, ParticlesPipeline, ScenePipeline};
use crate::swap_chain::SwapChain;

#[non_exhaustive]
pub struct SceneCommands {
    pub pool: vk::CommandPool,
    pub cmd_buffers: Vec<vk::CommandBuffer>,
}

impl SceneCommands {
    pub fn new(
        device: &Device,
        physical_device: &PhysicalDevice,
        max_frames_inflight: usize,
    ) -> VkResult<Self> {
        // Create command pool
        let pool_ci = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(physical_device.queue_family);
        let pool = unsafe { device.handle.create_command_pool(&pool_ci, None)? };

        // Allocate graphics command buffer
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(max_frames_inflight as u32);
        let cmd_buffers = unsafe { device.handle.allocate_command_buffers(&alloc_info)? };

        Ok(Self { pool, cmd_buffers })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &mut self,
        device: &Device,
        swap_chain: &SwapChain,
        pipeline: &ScenePipeline,
        vertex_buffer: &VertexBuffer,
        index_buffer: &IndexBuffer,
        game_objects: &[GameObject],
        image_index: usize,
        frame_index: usize,
    ) -> VkResult<()> {
        let device_h = &device.handle;
        let cmd_buffer = self.cmd_buffers[frame_index];

        let begin_info = vk::CommandBufferBeginInfo::default();
        unsafe { device_h.begin_command_buffer(cmd_buffer, &begin_info)? };

        // Transition swapchain image to color attachment optimal for rendering
        //
        // src_stage:  COLOR_ATTACHMENT_OUTPUT  (wait until this stage is done...)
        // src_access: empty                    (...with no prior writes to make visible)
        // dst_stage:  COLOR_ATTACHMENT_OUTPUT  (before this stage starts...)
        // dst_access: COLOR_ATTACHMENT_WRITE   (...writing color output)
        //
        // The barrier acts as a dividing line within the stage:
        // - Wait for any COLOR_ATTACHMENT_OUTPUT work that was submitted before this barrier
        // - Then do the transition
        // - Then allow COLOR_ATTACHMENT_OUTPUT work submitted after this barrier to proceed
        //
        // In this case there is no prior COLOR_ATTACHMENT_OUTPUT work at all — this is
        // the very start of the frame. So the src side resolves instantly (nothing to
        // wait for), the transition happens, and then the actual color writes are
        // unblocked.
        transition_image_layout(
            device_h,
            cmd_buffer,
            swap_chain.images[image_index],
            vk::ImageAspectFlags::COLOR,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT, // src_stage
            vk::AccessFlags2::empty(),                        // src_access_mask
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT, // dst_stage
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,         // dst_access_mask
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            1, // swapchain images always have a single mip level
        );

        // Transition the multisampled color image to color attachment optimal for rendering
        transition_image_layout(
            device_h,
            cmd_buffer,
            swap_chain.color_image.handle(),
            vk::ImageAspectFlags::COLOR,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            1,
        );

        // Transition the depth image to depth attachment optimal for rendering
        transition_image_layout(
            device_h,
            cmd_buffer,
            swap_chain.depth_image.handle(),
            vk::ImageAspectFlags::DEPTH,
            vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
            vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
            1,
        );

        // Color attachment (multisampled) with resolve attachment
        let color_attachment_info = vk::RenderingAttachmentInfo::default()
            .image_view(swap_chain.color_image.view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .resolve_mode(vk::ResolveModeFlags::AVERAGE)
            .resolve_image_view(swap_chain.image_views[image_index])
            .resolve_image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE) // keep after rendering so it can be presented
            .clear_value(vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
            });
        let depth_attachment_info = vk::RenderingAttachmentInfo::default()
            .image_view(swap_chain.depth_image.view)
            .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE) // don't save the depth buffer after rendering
            .clear_value(vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                },
            });

        let rendering_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: swap_chain.extent,
            })
            .layer_count(1)
            .color_attachments(slice::from_ref(&color_attachment_info))
            .depth_attachment(&depth_attachment_info);

        unsafe {
            device_h.cmd_begin_rendering(cmd_buffer, &rendering_info);
            device_h.cmd_bind_pipeline(
                cmd_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.handle,
            );
            device_h.cmd_set_viewport(
                cmd_buffer,
                0,
                &[vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: swap_chain.extent.width as f32,
                    height: swap_chain.extent.height as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                }],
            );
            device_h.cmd_set_scissor(
                cmd_buffer,
                0,
                &[vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: swap_chain.extent,
                }],
            );
            device_h.cmd_bind_vertex_buffers(cmd_buffer, 0, &[vertex_buffer.handle()], &[0]);
            device_h.cmd_bind_index_buffer(
                cmd_buffer,
                index_buffer.handle(),
                0,
                vk::IndexType::UINT32,
            );
            for game_object in game_objects {
                device_h.cmd_bind_descriptor_sets(
                    cmd_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipeline.layout,
                    0,
                    slice::from_ref(&game_object.descriptors[frame_index]),
                    &[],
                );
                device_h.cmd_draw_indexed(cmd_buffer, index_buffer.length, 1, 0, 0, 0);
            }
            device_h.cmd_draw_indexed(cmd_buffer, index_buffer.length, 1, 0, 0, 0);
            device_h.cmd_end_rendering(cmd_buffer);
        }

        // Transition swapchain image to present layout
        //
        // src_stage:  COLOR_ATTACHMENT_OUTPUT  (wait until color writes are done...)
        // src_access: COLOR_ATTACHMENT_WRITE   (...and make those writes visible)
        // dst_stage:  BOTTOM_OF_PIPE           (before the end of the pipeline...)
        // dst_access: empty                    (...no GPU reads needed, presentation engine handles it)
        transition_image_layout(
            device_h,
            cmd_buffer,
            swap_chain.images[image_index],
            vk::ImageAspectFlags::COLOR,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
            vk::AccessFlags2::empty(),
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::PRESENT_SRC_KHR,
            1, // swapchain images always have a single mip level
        );

        unsafe { device_h.end_command_buffer(cmd_buffer)? };

        Ok(())
    }

    /// # Safety
    ///
    /// - Must be called before the `ash::Device` that was used to create this
    ///   `SceneCommands` is destroyed.
    /// - The command buffer must not be in a pending state (i.e. not currently
    ///   being executed by the GPU).
    /// - Must be called at most once. Calling it more than once is undefined
    ///   behaviour as the underlying handles become invalid after the first call.
    pub unsafe fn destroy(&mut self, device: &Device) {
        unsafe {
            // Freeing the pool implicitly frees all command buffers allocated from it
            device.handle.destroy_command_pool(self.pool, None);
        }
    }
}

/// Allocates a one-time-submit command buffer, calls `f` to record into it,
/// then submits and waits for completion.
pub fn one_time_submit(
    device: &ash::Device,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    f: impl FnOnce(vk::CommandBuffer),
) -> VkResult<()> {
    // Allocate a short-lived command buffer
    let alloc_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let cmd_buffer = unsafe { device.allocate_command_buffers(&alloc_info)?[0] };

    let begin_info =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    unsafe { device.begin_command_buffer(cmd_buffer, &begin_info)? };

    f(cmd_buffer);

    unsafe { device.end_command_buffer(cmd_buffer)? };

    // Submit and wait
    let submit_info = vk::SubmitInfo::default().command_buffers(slice::from_ref(&cmd_buffer));
    unsafe {
        device.queue_submit(queue, slice::from_ref(&submit_info), vk::Fence::null())?;
        device.queue_wait_idle(queue)?;
        device.free_command_buffers(command_pool, &[cmd_buffer]);
    }

    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////

#[non_exhaustive]
pub struct ParticlesCommands {
    pub pool: vk::CommandPool,
    pub graphics_cmd_buffers: Vec<vk::CommandBuffer>,
    pub compute_cmd_buffers: Vec<vk::CommandBuffer>,
}

impl ParticlesCommands {
    pub fn new(
        device: &Device,
        physical_device: &PhysicalDevice,
        max_frames_inflight: usize,
    ) -> VkResult<Self> {
        let pool_ci = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(physical_device.queue_family);

        let pool = unsafe { device.handle.create_command_pool(&pool_ci, None)? };

        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(max_frames_inflight as u32 * 2);

        let graphics_cmd_buffers = unsafe { device.handle.allocate_command_buffers(&alloc_info)? };
        let compute_cmd_buffers = unsafe { device.handle.allocate_command_buffers(&alloc_info)? };

        Ok(Self {
            pool,
            graphics_cmd_buffers,
            compute_cmd_buffers,
        })
    }

    pub fn record_compute(
        &mut self,
        device: &Device,
        pipeline: &ParticlesComputePipeline,
        descriptors: &ParticlesDescriptors,
        frame_index: usize,
    ) -> VkResult<()> {
        let cmd = self.compute_cmd_buffers[frame_index];

        unsafe {
            device
                .handle
                .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
            device
                .handle
                .begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())?;

            device
                .handle
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline.handle);
            device.handle.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                pipeline.layout,
                0,
                slice::from_ref(&descriptors.desc_sets[frame_index]),
                &[],
            );
            device
                .handle
                .cmd_dispatch(cmd, Particle::COUNT as u32 / 256, 1, 1);

            device.handle.end_command_buffer(cmd)?;
        }

        Ok(())
    }

    pub fn record_graphics(
        &mut self,
        device: &Device,
        swap_chain: &SwapChain,
        pipeline: &ParticlesPipeline,
        storage_buffers: &StorageBuffers,
        image_index: usize,
        frame_index: usize,
    ) -> VkResult<()> {
        let cmd = self.graphics_cmd_buffers[frame_index];
        record_particles_graphics(
            cmd,
            device,
            swap_chain,
            pipeline,
            storage_buffers,
            image_index,
            frame_index,
        )
    }

    /// # Safety
    ///
    /// - Must be called before the `ash::Device` that was used to create this
    ///   `ParticlesCommands` is destroyed.
    /// - No command buffers from this pool may be in use by the GPU.
    /// - Must be called at most once.
    pub unsafe fn destroy(&mut self, device: &Device) {
        unsafe { device.handle.destroy_command_pool(self.pool, None) };
    }
}

fn record_particles_graphics(
    cmd: vk::CommandBuffer,
    device: &Device,
    swap_chain: &SwapChain,
    pipeline: &ParticlesPipeline,
    storage_buffers: &StorageBuffers,
    image_index: usize,
    frame_index: usize,
) -> VkResult<()> {
    let device_h = &device.handle;
    let swapchain_image = swap_chain.images[image_index];
    let extent = swap_chain.extent;

    unsafe {
        device_h.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
        device_h.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())?;

        transition_image_layout(
            device_h,
            cmd,
            swapchain_image,
            vk::ImageAspectFlags::COLOR,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags2::empty(),
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            1,
        );

        let clear_value = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        };

        let attachment_info = vk::RenderingAttachmentInfo::default()
            .image_view(swap_chain.image_views[image_index])
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(clear_value);

        let rendering_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent,
            })
            .layer_count(1)
            .color_attachments(slice::from_ref(&attachment_info));

        device_h.cmd_begin_rendering(cmd, &rendering_info);

        device_h.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline.handle);

        device_h.cmd_set_viewport(
            cmd,
            0,
            &[vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: extent.width as f32,
                height: extent.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            }],
        );

        device_h.cmd_set_scissor(
            cmd,
            0,
            &[vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent,
            }],
        );

        device_h.cmd_bind_vertex_buffers(
            cmd,
            0,
            &[storage_buffers.buffers[frame_index].handle()],
            &[0],
        );

        device_h.cmd_draw(cmd, Particle::COUNT as u32, 1, 0, 0);

        device_h.cmd_end_rendering(cmd);

        transition_image_layout(
            device_h,
            cmd,
            swapchain_image,
            vk::ImageAspectFlags::COLOR,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
            vk::AccessFlags2::empty(),
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::PRESENT_SRC_KHR,
            1,
        );

        device_h.end_command_buffer(cmd)?;
    }

    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////

#[non_exhaustive]
pub struct ParticlesMtCommands {
    pub pool: vk::CommandPool,
    pub graphics_cmd_buffers: Vec<vk::CommandBuffer>,
}

impl ParticlesMtCommands {
    pub fn new(
        device: &Device,
        physical_device: &PhysicalDevice,
        max_frames_inflight: usize,
    ) -> VkResult<Self> {
        let pool_ci = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(physical_device.queue_family);

        let pool = unsafe { device.handle.create_command_pool(&pool_ci, None)? };

        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(max_frames_inflight as u32);

        let graphics_cmd_buffers = unsafe { device.handle.allocate_command_buffers(&alloc_info)? };

        Ok(Self {
            pool,
            graphics_cmd_buffers,
        })
    }

    pub fn record_graphics(
        &mut self,
        device: &Device,
        swap_chain: &SwapChain,
        pipeline: &ParticlesPipeline,
        storage_buffers: &StorageBuffers,
        image_index: usize,
        frame_index: usize,
    ) -> VkResult<()> {
        let cmd = self.graphics_cmd_buffers[frame_index];
        record_particles_graphics(
            cmd,
            device,
            swap_chain,
            pipeline,
            storage_buffers,
            image_index,
            frame_index,
        )
    }

    /// # Safety
    ///
    /// - Must be called before the `ash::Device` that was used to create this
    ///   `ParticlesMtCommands` is destroyed.
    /// - No command buffers from this pool may be in use by the GPU.
    /// - Must be called at most once.
    pub unsafe fn destroy(&mut self, device: &Device) {
        unsafe { device.handle.destroy_command_pool(self.pool, None) };
    }
}
