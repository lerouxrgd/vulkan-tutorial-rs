use std::slice;

use ash::prelude::VkResult;
use ash::vk;

use crate::buffers::find_memory_type;
use crate::commands::one_time_submit;

#[non_exhaustive]
pub struct RawImage {
    pub handle: vk::Image,
    pub memory: vk::DeviceMemory,
}

impl RawImage {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        device: &ash::Device,
        width: u32,
        height: u32,
        format: vk::Format,
        tiling: vk::ImageTiling,
        usage: vk::ImageUsageFlags,
        properties: vk::MemoryPropertyFlags,
    ) -> anyhow::Result<Self> {
        let image_ci = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(
                // How many texels there are on each axis (width * height * depth)
                vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                },
            )
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(tiling)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let handle = unsafe { device.create_image(&image_ci, None)? };

        let mem_requirements = unsafe { device.get_image_memory_requirements(handle) };
        let memory_type_index = find_memory_type(
            instance,
            physical_device,
            mem_requirements.memory_type_bits,
            properties,
        )?;
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_requirements.size)
            .memory_type_index(memory_type_index);
        let memory = unsafe { device.allocate_memory(&alloc_info, None)? };

        unsafe { device.bind_image_memory(handle, memory, 0)? };

        Ok(Self { handle, memory })
    }

    /// # Safety
    ///
    /// - Must be called before the `ash::Device` that was used to create this
    ///   `RawImage` is destroyed.
    /// - The image must not be in use by the GPU.
    /// - Must be called at most once. Calling it more than once is undefined
    ///   behaviour as the underlying handles become invalid after the first call.
    pub unsafe fn destroy(&mut self, device: &ash::Device) {
        unsafe {
            device.destroy_image(self.handle, None);
            device.free_memory(self.memory, None);
        }
    }
}

/// Perform image layout transitions which tell the GPU how the image data is
/// physically arranged in memory.
#[allow(clippy::too_many_arguments)]
pub fn transition_image_layout(
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
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED) // no qf ownership transfer
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED) // no qf ownership transfer
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

pub fn copy_buffer_to_image(
    device: &ash::Device,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    buffer: vk::Buffer,
    image: vk::Image,
    width: u32,
    height: u32,
) -> VkResult<()> {
    one_time_submit(device, queue, command_pool, |cmd_buffer| {
        let region = vk::BufferImageCopy::default()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .mip_level(0)
                    .base_array_layer(0)
                    .layer_count(1),
            )
            .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .image_extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            });

        unsafe {
            device.cmd_copy_buffer_to_image(
                cmd_buffer,
                buffer,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                slice::from_ref(&region),
            );
        }
    })
}
