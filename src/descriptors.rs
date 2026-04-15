use std::mem;
use std::slice;

use ash::prelude::VkResult;
use ash::vk;

use crate::buffers::{UniformBufferObject, UniformBuffers};
use crate::devices::Device;
use crate::pipeline::GraphicsPipeline;

#[non_exhaustive]
pub struct UboDescriptors {
    pub pool: vk::DescriptorPool,
    pub desc_sets: Vec<vk::DescriptorSet>,
}

impl UboDescriptors {
    pub fn new(device: &Device, max_frames_in_flight: usize) -> VkResult<Self> {
        let pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(max_frames_in_flight as u32);

        let pool_ci = vk::DescriptorPoolCreateInfo::default()
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
            .max_sets(max_frames_in_flight as u32)
            .pool_sizes(slice::from_ref(&pool_size));

        let pool = unsafe { device.handle.create_descriptor_pool(&pool_ci, None)? };

        Ok(Self {
            pool,
            desc_sets: Vec::new(),
        })
    }

    pub fn allocate_ubo_desc_sets(
        &mut self,
        device: &Device,
        pipeline: &GraphicsPipeline,
        uniform_buffers: &UniformBuffers,
    ) -> VkResult<()> {
        let max_frames_in_flight = uniform_buffers.len();

        let layouts = vec![pipeline.descriptor_set_layout; max_frames_in_flight];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.pool)
            .set_layouts(&layouts);

        self.desc_sets = unsafe { device.handle.allocate_descriptor_sets(&alloc_info)? };

        for (i, &desc_set) in self.desc_sets.iter().enumerate() {
            let buffer_info = vk::DescriptorBufferInfo::default()
                .buffer(uniform_buffers.buffers[i].handle())
                .offset(0)
                .range(mem::size_of::<UniformBufferObject>() as vk::DeviceSize);

            // For descriptor set i, at binding 0, point at uniform_buffers[i] starting
            // at offset 0 with a range of sizeof(UniformBufferObject)
            let descriptor_write = vk::WriteDescriptorSet::default()
                .dst_set(desc_set)
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(slice::from_ref(&buffer_info));

            unsafe {
                // Wires each descriptor set to its corresponding vk::Buffer handle
                device
                    .handle
                    .update_descriptor_sets(slice::from_ref(&descriptor_write), &[])
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
        }
    }
}
