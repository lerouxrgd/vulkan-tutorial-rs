use std::mem;
use std::slice;

use anyhow::ensure;
use ash::prelude::VkResult;
use ash::vk;

use crate::buffers::{DeltaTime, MvpMatrices, Particle, StorageBuffers, UniformBuffers};
use crate::devices::Device;
use crate::images::{TextureImage, TextureSampler};

#[non_exhaustive]
pub struct SceneDescriptorPool {
    pub desc_set_layout: vk::DescriptorSetLayout,
    pub pool: vk::DescriptorPool,
    pub max_frames_inflight: usize,
}

impl SceneDescriptorPool {
    pub fn new(device: &Device, max_frames_inflight: usize, max_objects: usize) -> VkResult<Self> {
        let total = (max_frames_inflight * max_objects) as u32;

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

        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(total),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(total),
        ];
        let pool_ci = vk::DescriptorPoolCreateInfo::default()
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
            .max_sets(total)
            .pool_sizes(&pool_sizes);
        let pool = unsafe { device.handle.create_descriptor_pool(&pool_ci, None)? };

        Ok(Self {
            desc_set_layout,
            pool,
            max_frames_inflight,
        })
    }

    pub fn allocate(
        &self,
        device: &Device,
        uniform_buffers: &UniformBuffers<MvpMatrices>,
        texture: &TextureImage,
        sampler: &TextureSampler,
    ) -> anyhow::Result<SceneDescriptorSets> {
        ensure!(uniform_buffers.len() == self.max_frames_inflight);

        let layouts = vec![self.desc_set_layout; self.max_frames_inflight];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.pool)
            .set_layouts(&layouts);

        let desc_sets = unsafe { device.handle.allocate_descriptor_sets(&alloc_info)? };

        for (i, &desc_set) in desc_sets.iter().enumerate() {
            let buffer_info = vk::DescriptorBufferInfo::default()
                .buffer(uniform_buffers.buffers[i].handle())
                .offset(0)
                .range(uniform_buffers.ubo_size() as vk::DeviceSize);

            let image_info = vk::DescriptorImageInfo::default()
                .sampler(sampler.handle)
                .image_view(texture.view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

            let descriptor_writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(desc_set)
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(slice::from_ref(&buffer_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(desc_set)
                    .dst_binding(1)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(slice::from_ref(&image_info)),
            ];

            unsafe {
                device
                    .handle
                    .update_descriptor_sets(&descriptor_writes, &[])
            };
        }

        Ok(SceneDescriptorSets { desc_sets })
    }

    /// # Safety
    ///
    /// - Must be called after all `SceneDescriptorSets` allocated from this
    ///   pool have been freed or are no longer in use by the GPU.
    /// - Must be called at most once.
    pub unsafe fn destroy(&mut self, device: &Device) {
        unsafe {
            device.handle.destroy_descriptor_pool(self.pool, None);
            device
                .handle
                .destroy_descriptor_set_layout(self.desc_set_layout, None);
        }
    }
}

pub struct SceneDescriptorSets {
    desc_sets: Vec<vk::DescriptorSet>,
}

impl std::ops::Index<usize> for SceneDescriptorSets {
    type Output = vk::DescriptorSet;

    fn index(&self, index: usize) -> &Self::Output {
        self.desc_sets.index(index)
    }
}

////////////////////////////////////////////////////////////////////////////////////////

#[non_exhaustive]
pub struct ParticlesDescriptors {
    pub desc_set_layout: vk::DescriptorSetLayout,
    pub pool: vk::DescriptorPool,
    pub desc_sets: Vec<vk::DescriptorSet>,
}

impl ParticlesDescriptors {
    pub fn new(device: &Device, max_frames_in_flight: usize) -> VkResult<Self> {
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];

        let desc_set_layout_ci = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let desc_set_layout = unsafe {
            device
                .handle
                .create_descriptor_set_layout(&desc_set_layout_ci, None)?
        };

        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(max_frames_in_flight as u32),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(max_frames_in_flight as u32 * 2), // last + current frame
        ];

        let pool_ci = vk::DescriptorPoolCreateInfo::default()
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
            .max_sets(max_frames_in_flight as u32)
            .pool_sizes(&pool_sizes);

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
        uniform_buffers: &UniformBuffers<DeltaTime>,
        storage_buffers: &StorageBuffers,
    ) -> VkResult<()> {
        let max_frames_in_flight = uniform_buffers.len();

        let layouts = vec![self.desc_set_layout; max_frames_in_flight];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.pool)
            .set_layouts(&layouts);

        self.desc_sets = unsafe { device.handle.allocate_descriptor_sets(&alloc_info)? };

        for (i, &desc_set) in self.desc_sets.iter().enumerate() {
            let ubo_info = vk::DescriptorBufferInfo::default()
                .buffer(uniform_buffers.buffers[i].handle())
                .offset(0)
                .range(uniform_buffers.ubo_size() as vk::DeviceSize);

            let last_frame = (i + max_frames_in_flight - 1) % max_frames_in_flight;
            let storage_last_frame_info = vk::DescriptorBufferInfo::default()
                .buffer(storage_buffers.buffers[last_frame].handle())
                .offset(0)
                .range((mem::size_of::<Particle>() * Particle::COUNT) as vk::DeviceSize);

            let storage_current_frame_info = vk::DescriptorBufferInfo::default()
                .buffer(storage_buffers.buffers[i].handle())
                .offset(0)
                .range((mem::size_of::<Particle>() * Particle::COUNT) as vk::DeviceSize);

            let descriptor_writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(desc_set)
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(slice::from_ref(&ubo_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(desc_set)
                    .dst_binding(1)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(slice::from_ref(&storage_last_frame_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(desc_set)
                    .dst_binding(2)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(slice::from_ref(&storage_current_frame_info)),
            ];

            unsafe {
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
    ///   `ParticlesDescriptors` is destroyed.
    /// - All descriptor sets allocated from the pool must no longer be in use
    ///   by the GPU.
    /// - Must be called at most once.
    pub unsafe fn destroy(&mut self, device: &Device) {
        unsafe {
            device.handle.destroy_descriptor_pool(self.pool, None);
            device
                .handle
                .destroy_descriptor_set_layout(self.desc_set_layout, None);
        }
    }
}
