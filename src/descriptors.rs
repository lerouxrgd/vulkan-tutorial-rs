use std::mem;
use std::slice;

use ash::prelude::VkResult;
use ash::vk;

use crate::buffers::{UniformBufferObject, UniformBuffers};
use crate::devices::Device;
use crate::images::{TextureImage, TextureSampler};

#[non_exhaustive]
pub struct Descriptors {
    pub desc_set_layout: vk::DescriptorSetLayout,
    pub pool: vk::DescriptorPool,
    pub desc_sets: Vec<vk::DescriptorSet>,
}

impl Descriptors {
    pub fn new(device: &Device, max_frames_in_flight: usize) -> VkResult<Self> {
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::VERTEX),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        ];
        let desc_set_layout_ci = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let desc_set_layout = unsafe {
            device
                .handle
                .create_descriptor_set_layout(&desc_set_layout_ci, None)?
        };

        let pool_size = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(max_frames_in_flight as u32),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(max_frames_in_flight as u32),
        ];

        let pool_ci = vk::DescriptorPoolCreateInfo::default()
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
            .max_sets(max_frames_in_flight as u32)
            .pool_sizes(&pool_size);

        let pool = unsafe { device.handle.create_descriptor_pool(&pool_ci, None)? };

        Ok(Self {
            desc_set_layout,
            pool,
            desc_sets: Vec::new(),
        })
    }

    pub fn allocate_desc_sets(
        &mut self,
        device: &Device,
        uniform_buffers: &UniformBuffers,
        texture: &TextureImage,
        sampler: &TextureSampler,
    ) -> VkResult<()> {
        let max_frames_in_flight = uniform_buffers.len();

        let layouts = vec![self.desc_set_layout; max_frames_in_flight];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.pool)
            .set_layouts(&layouts);

        self.desc_sets = unsafe { device.handle.allocate_descriptor_sets(&alloc_info)? };

        for (i, &desc_set) in self.desc_sets.iter().enumerate() {
            let buffer_info = vk::DescriptorBufferInfo::default()
                .buffer(uniform_buffers.buffers[i].handle())
                .offset(0)
                .range(mem::size_of::<UniformBufferObject>() as vk::DeviceSize);

            let image_info = vk::DescriptorImageInfo::default()
                .sampler(sampler.handle)
                .image_view(texture.view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

            // For descriptor set i...
            let descriptor_writes = [
                // At binding 0, point at uniform_buffers[i] starting at offset 0
                // with a range of sizeof(UniformBufferObject)
                vk::WriteDescriptorSet::default()
                    .dst_set(desc_set)
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(slice::from_ref(&buffer_info)),
                // At binding 1, point at combined image/sampler
                vk::WriteDescriptorSet::default()
                    .dst_set(desc_set)
                    .dst_binding(1)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(slice::from_ref(&image_info)),
            ];

            unsafe {
                // Wires each descriptor set to its corresponding data (vk::Buffer handle, image/sampler)
                device
                    .handle
                    .update_descriptor_sets(&descriptor_writes, &[])
            };
        }

        Ok(())
    }

    /// # Safety
    ///
    /// - Must be called before the `ash::Device` that was used to create this
    ///   `Descriptor` is destroyed.
    /// - All descriptor sets allocated from the pool must no longer be in use
    ///   by the GPU.
    /// - Must be called at most once. Calling it more than once is undefined
    ///   behaviour as the underlying handles become invalid after the first call.
    pub unsafe fn destroy(&mut self, device: &Device) {
        unsafe {
            // Freeing the pool implicitly frees all descriptor sets allocated from it
            device.handle.destroy_descriptor_pool(self.pool, None);
            device
                .handle
                .destroy_descriptor_set_layout(self.desc_set_layout, None);
        }
    }
}
