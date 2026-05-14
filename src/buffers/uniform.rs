use std::mem;
use std::time::Duration;

use ash::vk;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};

use crate::buffers::raw::RawBuffer;
use crate::devices::{Device, PhysicalDevice};
use crate::instance::Instance;
use crate::swap_chain::SwapChain;

#[non_exhaustive]
pub struct UniformBuffer<T> {
    raw: RawBuffer,
    pub mapped: *mut T,
}

impl<T> UniformBuffer<T>
where
    T: Copy,
{
    pub fn new(
        instance: &Instance,
        physical_device: &PhysicalDevice,
        device: &Device,
    ) -> anyhow::Result<Self> {
        let instance_h = &instance.handle;
        let physical_device_h = physical_device.handle;
        let device_h = &device.handle;

        let size = mem::size_of::<T>() as vk::DeviceSize;
        let raw = RawBuffer::new(
            instance_h,
            physical_device_h,
            device_h,
            size,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let mapped = unsafe {
            device_h.map_memory(raw.memory, 0, size, vk::MemoryMapFlags::empty())? as *mut T
        };

        Ok(Self { raw, mapped })
    }

    pub fn handle(&self) -> vk::Buffer {
        self.raw.handle
    }

    pub fn write(&mut self, value: T) {
        unsafe { self.mapped.write(value) };
    }

    pub unsafe fn destroy(&mut self, device: &Device) {
        unsafe {
            device.handle.unmap_memory(self.raw.memory);
            self.raw.destroy(&device.handle);
        }
    }
}

pub struct UniformBuffers<T> {
    pub buffers: Vec<UniformBuffer<T>>,
}

impl<T> UniformBuffers<T>
where
    T: Copy,
{
    pub fn new(
        instance: &Instance,
        physical_device: &PhysicalDevice,
        device: &Device,
        max_frames_in_flight: usize,
    ) -> anyhow::Result<Self> {
        let buffers = (0..max_frames_in_flight)
            .map(|_| UniformBuffer::new(instance, physical_device, device))
            .collect::<anyhow::Result<Vec<_>>>()?;
        Ok(Self { buffers })
    }

    pub fn len(&self) -> usize {
        self.buffers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn ubo_size(&self) -> usize {
        mem::size_of::<T>()
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
        for uniform_buffer in self.buffers.iter_mut() {
            unsafe {
                uniform_buffer.destroy(device);
            }
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct MvpMatrices {
    pub model: Mat4,
    pub view: Mat4,
    pub proj: Mat4,
}

impl UniformBuffers<MvpMatrices> {
    pub fn update(&mut self, current_frame: usize, delta_time: f32, swap_chain: &SwapChain) {
        let extent = &swap_chain.extent;

        let model = Mat4::from_rotation_z(delta_time * 90.0_f32.to_radians());
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

        let ubo = MvpMatrices { model, view, proj };
        self.buffers[current_frame].write(ubo);
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct DeltaTime {
    pub delta_time: f32,
}

impl UniformBuffers<DeltaTime> {
    pub fn update(&mut self, frame: usize, delta_time: Duration) {
        self.buffers[frame].write(DeltaTime {
            delta_time: delta_time.as_secs_f32() * 1000.0 * 2.0,
        });
    }
}
