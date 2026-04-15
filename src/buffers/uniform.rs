use std::mem;

use ash::vk;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};

use crate::buffers::raw::RawBuffer;
use crate::devices::{Device, PhysicalDevice};
use crate::instance::Instance;
use crate::swap_chain::SwapChain;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct UniformBufferObject {
    pub model: Mat4,
    pub view: Mat4,
    pub proj: Mat4,
}

#[non_exhaustive]
pub struct UniformBuffer {
    raw: RawBuffer,
    pub mapped: *mut UniformBufferObject,
}

impl UniformBuffer {
    pub fn handle(&self) -> vk::Buffer {
        self.raw.handle
    }
}

pub struct UniformBuffers {
    pub buffers: Vec<UniformBuffer>,
}

impl UniformBuffers {
    pub fn new(
        instance: &Instance,
        physical_device: &PhysicalDevice,
        device: &Device,
        max_frames_in_flight: usize,
    ) -> anyhow::Result<Self> {
        let instance_h = &instance.handle;
        let physical_device_h = physical_device.handle;
        let device_h = &device.handle;

        let size = mem::size_of::<UniformBufferObject>() as vk::DeviceSize;
        let buffers = (0..max_frames_in_flight)
            .map(|_| {
                let raw = RawBuffer::new(
                    instance_h,
                    physical_device_h,
                    device_h,
                    size,
                    vk::BufferUsageFlags::UNIFORM_BUFFER,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                )?;
                let mapped = unsafe {
                    device_h.map_memory(raw.memory, 0, size, vk::MemoryMapFlags::empty())?
                        as *mut UniformBufferObject
                };
                Ok(UniformBuffer { raw, mapped })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        Ok(Self { buffers })
    }

    pub fn len(&self) -> usize {
        self.buffers.len()
    }

    pub fn update(&self, current_frame: usize, time_delta: f32, swap_chain: &SwapChain) {
        let extent = &swap_chain.extent;

        let model = Mat4::from_rotation_z(time_delta * 90.0_f32.to_radians());
        let view = Mat4::look_at_rh(
            Vec3::new(2.0, 2.0, 2.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        );
        let mut proj = Mat4::perspective_rh(
            45.0_f32.to_radians(),
            extent.width as f32 / extent.height as f32,
            0.1,
            10.0,
        );
        proj.y_axis.y *= -1.0; // flip Y axis: glm uses OpenGL convention (Y up), Vulkan has Y down in NDC

        let ubo = UniformBufferObject { model, view, proj };

        unsafe {
            self.buffers[current_frame].mapped.write(ubo);
        }
    }
    /// # Safety
    ///
    /// - Must be called before the `ash::Device` that was used to create this
    ///   `UniformBuffers` is destroyed.
    /// - No GPU commands may be reading from any of the uniform buffers when
    ///   this is called.
    /// - Must be called at most once. Calling it more than once is undefined
    ///   behaviour as the underlying handles become invalid after the first call.
    pub unsafe fn destroy(&mut self, device: &Device) {
        for buffer in &self.buffers {
            unsafe {
                device.handle.unmap_memory(buffer.raw.memory);
            }
        }
        for mut buffer in self.buffers.drain(..) {
            unsafe { buffer.raw.destroy(&device.handle) };
        }
    }
}
