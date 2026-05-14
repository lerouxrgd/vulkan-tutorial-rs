use std::path::Path;
use std::slice;

use anyhow::ensure;
use ash::prelude::VkResult;
use ash::vk::{self, ImageUsageFlags};

use crate::buffers::RawBuffer;
use crate::commands::{SceneCommands, one_time_submit};
use crate::devices::{Device, PhysicalDevice};
use crate::images::raw::{RawImage, copy_buffer_to_image, transition_image_layout};
use crate::instance::Instance;

#[non_exhaustive]
pub struct TextureImage {
    raw: RawImage,
    pub view: vk::ImageView,
}

impl TextureImage {
    pub fn new<P>(
        instance: &Instance,
        physical_device: &PhysicalDevice,
        device: &Device,
        commands: &SceneCommands,
        path: P,
    ) -> anyhow::Result<Self>
    where
        P: AsRef<Path>,
    {
        let device_h = &device.handle;

        // Load image from disk
        let img = image::open(path)?.into_rgba8();
        let (width, height) = img.dimensions();
        let mip_levels = (width.max(height) as f32).log2().floor() as u32 + 1;
        let pixels = img.into_raw();
        let image_size = (width * height * 4) as vk::DeviceSize;

        // Upload pixels to staging buffer
        let mut staging = RawBuffer::new(
            &instance.handle,
            physical_device.handle,
            device_h,
            image_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        unsafe {
            let data =
                device_h.map_memory(staging.memory, 0, image_size, vk::MemoryMapFlags::empty())?;
            let slice = slice::from_raw_parts_mut(data as *mut u8, image_size as usize);
            slice.copy_from_slice(&pixels);
            device_h.unmap_memory(staging.memory);
        }

        // Create device-local image
        let raw = RawImage::new(
            &instance.handle,
            physical_device.handle,
            &device.handle,
            width,
            height,
            vk::Format::R8G8B8A8_SRGB,
            vk::ImageTiling::OPTIMAL,
            ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::SAMPLED,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            mip_levels,
            vk::SampleCountFlags::TYPE_1,
        )?;

        one_time_submit(device_h, device.queue, commands.pool, |cmd| {
            transition_image_layout(
                device_h,
                cmd,
                raw.handle,
                vk::ImageAspectFlags::COLOR,
                vk::PipelineStageFlags2::TOP_OF_PIPE,
                vk::AccessFlags2::empty(),
                vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::TRANSFER_WRITE,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                mip_levels,
            );
        })?;

        copy_buffer_to_image(
            device_h,
            device.queue,
            commands.pool,
            staging.handle, // src
            raw.handle,     // dst
            width,
            height,
        )?;

        generate_mipmaps(
            &instance.handle,
            physical_device.handle,
            device_h,
            device.queue,
            commands.pool,
            raw.handle,
            vk::Format::R8G8B8A8_SRGB,
            width,
            height,
            mip_levels,
        )?;

        unsafe { staging.destroy(device_h) };

        // Create image view
        let view_ci = vk::ImageViewCreateInfo::default()
            .image(raw.handle)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_SRGB)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(mip_levels)
                    .base_array_layer(0)
                    .layer_count(1),
            );
        let view = unsafe { device_h.create_image_view(&view_ci, None)? };

        Ok(Self { raw, view })
    }

    /// # Safety
    ///
    /// - Must be called before the `ash::Device` that was used to create this
    ///   `TextureImage` is destroyed.
    /// - The image must not be in use by the GPU.
    /// - Must be called at most once. Calling it more than once is undefined
    ///   behaviour as the underlying handles become invalid after the first call.
    pub unsafe fn destroy(&mut self, device: &Device) {
        unsafe {
            device.handle.destroy_image_view(self.view, None);
            self.raw.destroy(&device.handle);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn generate_mipmaps(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    device: &ash::Device,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    image: vk::Image,
    format: vk::Format,
    width: u32,
    height: u32,
    mip_levels: u32,
) -> anyhow::Result<()> {
    log::debug!("Generating {mip_levels} mipmaps");

    // Check linear blit support
    let format_properties =
        unsafe { instance.get_physical_device_format_properties(physical_device, format) };
    ensure!(
        format_properties
            .optimal_tiling_features
            .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR),
        "texture image format does not support linear blitting"
    );

    one_time_submit(device, queue, command_pool, |cmd| {
        let mut barrier = vk::ImageMemoryBarrier2::default()
            .image(image)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
            .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            );

        let mut mip_width = width as i32;
        let mut mip_height = height as i32;

        for i in 1..mip_levels {
            barrier = barrier
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(i - 1)
                        .level_count(1)
                        .base_array_layer(0)
                        .layer_count(1),
                )
                .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL);

            let dependency_info =
                vk::DependencyInfo::default().image_memory_barriers(slice::from_ref(&barrier));
            unsafe { device.cmd_pipeline_barrier2(cmd, &dependency_info) };

            let blit = vk::ImageBlit::default()
                .src_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .mip_level(i - 1)
                        .base_array_layer(0)
                        .layer_count(1),
                )
                .src_offsets([
                    vk::Offset3D::default(),
                    vk::Offset3D {
                        x: mip_width,
                        y: mip_height,
                        z: 1,
                    },
                ])
                .dst_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .mip_level(i)
                        .base_array_layer(0)
                        .layer_count(1),
                )
                .dst_offsets([
                    vk::Offset3D::default(),
                    vk::Offset3D {
                        x: if mip_width > 1 { mip_width / 2 } else { 1 },
                        y: if mip_height > 1 { mip_height / 2 } else { 1 },
                        z: 1,
                    },
                ]);

            unsafe {
                device.cmd_blit_image(
                    cmd,
                    image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    slice::from_ref(&blit),
                    vk::Filter::LINEAR, // enables interpolation
                )
            };

            barrier = barrier
                .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .src_access_mask(vk::AccessFlags2::TRANSFER_READ)
                .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

            let dependency_info =
                vk::DependencyInfo::default().image_memory_barriers(slice::from_ref(&barrier));
            unsafe { device.cmd_pipeline_barrier2(cmd, &dependency_info) };

            if mip_width > 1 {
                mip_width /= 2;
            }
            if mip_height > 1 {
                mip_height /= 2;
            }
        }

        // Transition the last mip level
        barrier = barrier
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(mip_levels - 1)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            )
            .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
            .dst_access_mask(vk::AccessFlags2::SHADER_READ)
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

        let dependency_info =
            vk::DependencyInfo::default().image_memory_barriers(slice::from_ref(&barrier));
        unsafe { device.cmd_pipeline_barrier2(cmd, &dependency_info) };
    })?;

    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////

#[non_exhaustive]
pub struct TextureSampler {
    pub handle: vk::Sampler,
}

impl TextureSampler {
    pub fn new(
        instance: &Instance,
        physical_device: &PhysicalDevice,
        device: &Device,
    ) -> VkResult<Self> {
        let properties = unsafe {
            instance
                .handle
                .get_physical_device_properties(physical_device.handle)
        };

        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::REPEAT)
            .address_mode_w(vk::SamplerAddressMode::REPEAT)
            .anisotropy_enable(true)
            .max_anisotropy(properties.limits.max_sampler_anisotropy)
            .compare_enable(false)
            .compare_op(vk::CompareOp::ALWAYS)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .mip_lod_bias(0.0)
            .min_lod(0.0)
            .max_lod(vk::LOD_CLAMP_NONE);
        let handle = unsafe { device.handle.create_sampler(&sampler_info, None) }?;

        Ok(Self { handle })
    }

    /// # Safety
    ///
    /// - Must be called before the `ash::Device` that was used to create this
    ///   `TextureSampler` is destroyed.
    /// - The image must not be in use by the GPU.
    /// - Must be called at most once. Calling it more than once is undefined
    ///   behaviour as the underlying handles become invalid after the first call.
    pub unsafe fn destroy(&mut self, device: &Device) {
        unsafe {
            device.handle.destroy_sampler(self.handle, None);
        }
    }
}
