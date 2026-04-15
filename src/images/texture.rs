use std::path::Path;
use std::slice;

use ash::vk;

use crate::buffers::RawBuffer;
use crate::commands::{Commands, one_time_submit};
use crate::devices::{Device, PhysicalDevice};
use crate::images::raw::{RawImage, copy_buffer_to_image, transition_image_layout};
use crate::instance::Instance;

#[non_exhaustive]
pub struct TextureImage {
    pub image: RawImage,
    pub view: vk::ImageView,
}

impl TextureImage {
    pub fn new<P>(
        instance: &Instance,
        physical_device: &PhysicalDevice,
        device: &Device,
        commands: &Commands,
        path: P,
    ) -> anyhow::Result<Self>
    where
        P: AsRef<Path>,
    {
        let device_h = &device.handle;

        // Load image from disk
        let img = image::open(path)?.into_rgba8();
        let (width, height) = img.dimensions();
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
        let image = RawImage::new(
            &instance.handle,
            physical_device.handle,
            &device.handle,
            width,
            height,
            vk::Format::R8G8B8A8_SRGB,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        // Transition + copy + transition, using a one-time command buffer each time
        one_time_submit(device_h, device.queue, commands.pool, |cmd| {
            transition_image_layout(
                device_h,
                cmd,
                image.handle,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::AccessFlags2::empty(),
                vk::AccessFlags2::TRANSFER_WRITE,
                vk::PipelineStageFlags2::TOP_OF_PIPE,
                vk::PipelineStageFlags2::TRANSFER,
            );
        })?;

        copy_buffer_to_image(
            device_h,
            device.queue,
            commands.pool,
            staging.handle, // src
            image.handle,   // dst
            width,
            height,
        )?;

        one_time_submit(device_h, device.queue, commands.pool, |cmd| {
            transition_image_layout(
                device_h,
                cmd,
                image.handle,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::AccessFlags2::TRANSFER_WRITE,
                vk::AccessFlags2::SHADER_READ,
                vk::PipelineStageFlags2::TRANSFER,
                vk::PipelineStageFlags2::FRAGMENT_SHADER,
            );
        })?;

        unsafe { staging.destroy(device_h) };

        // Create image view
        let view_ci = vk::ImageViewCreateInfo::default()
            .image(image.handle)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_SRGB)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            );
        let view = unsafe { device_h.create_image_view(&view_ci, None)? };

        Ok(Self { image, view })
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
            self.image.destroy(&device.handle);
        }
    }
}
