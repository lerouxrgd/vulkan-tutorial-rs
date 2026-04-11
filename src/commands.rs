use std::slice;

use ash::prelude::VkResult;
use ash::vk;

use crate::buffers::{IndexBuffer, VertexBuffer};
use crate::descriptors::UboDescriptors;
use crate::devices::{Device, PhysicalDevice};
use crate::pipeline::GraphicsPipeline;
use crate::swap_chain::SwapChain;

#[non_exhaustive]
pub struct Commands {
    pub pool: vk::CommandPool,
    pub buffers: Vec<vk::CommandBuffer>,
}

impl Commands {
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

        // Allocate command buffer
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(max_frames_inflight as u32);
        let buffers = unsafe { device.handle.allocate_command_buffers(&alloc_info)? };

        Ok(Self { pool, buffers })
    }

    pub fn record(
        &mut self,
        device: &Device,
        swap_chain: &SwapChain,
        pipeline: &GraphicsPipeline,
        vertex_buffer: &VertexBuffer,
        index_buffer: &IndexBuffer,
        ubo_descriptors: &UboDescriptors,
        image_index: usize,
        frame_index: usize,
    ) -> VkResult<()> {
        let device_h = &device.handle;
        let cmd_buffer = self.buffers[frame_index];

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
        Self::transition_image_layout(
            device_h,
            cmd_buffer,
            swap_chain.images[image_index],
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::AccessFlags2::empty(),                // srcAccessMask
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE, // dstAccessMask
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT, // srcStage
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT, // dstStage
        );

        let attachment_info = vk::RenderingAttachmentInfo::default()
            .image_view(swap_chain.image_views[image_index])
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
            });

        let rendering_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: swap_chain.extent,
            })
            .layer_count(1)
            .color_attachments(slice::from_ref(&attachment_info));

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
            device_h.cmd_bind_vertex_buffers(cmd_buffer, 0, &[vertex_buffer.handle], &[0]);
            device_h.cmd_bind_index_buffer(
                cmd_buffer,
                index_buffer.handle,
                0,
                vk::IndexType::UINT16,
            );
            device_h.cmd_bind_descriptor_sets(
                cmd_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.layout,
                0,
                slice::from_ref(&ubo_descriptors.desc_sets[frame_index]),
                &[],
            );
            device_h.cmd_draw_indexed(cmd_buffer, index_buffer.len(), 1, 0, 0, 0);
            device_h.cmd_end_rendering(cmd_buffer);
        }

        // Transition swapchain image to present layout
        //
        // src_stage:  COLOR_ATTACHMENT_OUTPUT  (wait until color writes are done...)
        // src_access: COLOR_ATTACHMENT_WRITE   (...and make those writes visible)
        // dst_stage:  BOTTOM_OF_PIPE           (before the end of the pipeline...)
        // dst_access: empty                    (...no GPU reads needed, presentation engine handles it)
        Self::transition_image_layout(
            device_h,
            cmd_buffer,
            swap_chain.images[image_index],
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::PRESENT_SRC_KHR,
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            vk::AccessFlags2::empty(),
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
        );

        unsafe { device_h.end_command_buffer(cmd_buffer)? };

        Ok(())
    }

    /// Perform image layout transitions which tell the GPU how the image data is
    /// physically arranged in memory.
    #[allow(clippy::too_many_arguments)]
    fn transition_image_layout(
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        image: vk::Image,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
        src_access_mask: vk::AccessFlags2,
        dst_access_mask: vk::AccessFlags2,
        src_stage_mask: vk::PipelineStageFlags2,
        dst_stage_mask: vk::PipelineStageFlags2,
    ) {
        // Memory barriers (ImageMemoryBarrier2) synchronize within a single queue,
        // controlling both execution order and memory visibility between pipeline
        // stages.
        let barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(src_stage_mask)
            .src_access_mask(src_access_mask)
            .dst_stage_mask(dst_stage_mask)
            .dst_access_mask(dst_access_mask)
            .old_layout(old_layout)
            .new_layout(new_layout)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            );

        let dependency_info =
            vk::DependencyInfo::default().image_memory_barriers(slice::from_ref(&barrier));

        unsafe { device.cmd_pipeline_barrier2(command_buffer, &dependency_info) };
    }

    /// # Safety
    ///
    /// - Must be called before the `ash::Device` that was used to create this
    ///   `Commands` is destroyed.
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
