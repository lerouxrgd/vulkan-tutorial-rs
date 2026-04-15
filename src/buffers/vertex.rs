use std::{mem, slice};

use ash::vk;
use bytemuck::{Pod, Zeroable};
use glam::{Vec2, Vec3};

use crate::buffers::raw::RawBuffer;
use crate::commands::{Commands, one_time_submit};
use crate::devices::{Device, PhysicalDevice};
use crate::instance::Instance;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Vertex {
    pub pos: Vec2,
    pub color: Vec3,
}

impl Vertex {
    pub fn binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(mem::size_of::<Self>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
    }

    pub fn attribute_descriptions() -> [vk::VertexInputAttributeDescription; 2] {
        [
            vk::VertexInputAttributeDescription::default()
                .location(0)
                .binding(0)
                .format(vk::Format::R32G32_SFLOAT) // 2×f32
                .offset(mem::offset_of!(Self, pos) as u32),
            vk::VertexInputAttributeDescription::default()
                .location(1)
                .binding(0)
                .format(vk::Format::R32G32B32_SFLOAT) // 3×f32
                .offset(mem::offset_of!(Self, color) as u32),
        ]
    }
}

#[non_exhaustive]
pub struct VertexBuffer {
    raw: RawBuffer,
}

impl VertexBuffer {
    #[rustfmt::skip]
    pub const VERTICES: &[Vertex] = &[
        Vertex { pos: Vec2::new(-0.5, -0.5), color: Vec3::new(1.0, 0.0, 0.0) },
        Vertex { pos: Vec2::new(0.5, -0.5),  color: Vec3::new(0.0, 1.0, 0.0) },
        Vertex { pos: Vec2::new(0.5, 0.5),   color: Vec3::new(0.0, 0.0, 1.0) },
        Vertex { pos: Vec2::new(-0.5, 0.5),  color: Vec3::new(1.0, 1.0, 1.0) },
    ];

    pub fn new(
        instance: &Instance,
        physical_device: &PhysicalDevice,
        device: &Device,
        commands: &Commands,
    ) -> anyhow::Result<Self> {
        let instance_h = &instance.handle;
        let physical_device_h = physical_device.handle;
        let device_h = &device.handle;

        let size = mem::size_of_val(Self::VERTICES) as vk::DeviceSize;

        // Create staging buffer (CPU visible)
        let mut staging = RawBuffer::new(
            instance_h,
            physical_device_h,
            device_h,
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        // Copy vertex data into staging buffer
        unsafe {
            let data = device_h.map_memory(staging.memory, 0, size, vk::MemoryMapFlags::empty())?;
            let slice = slice::from_raw_parts_mut(data as *mut u8, size as usize);
            slice.copy_from_slice(bytemuck::cast_slice(Self::VERTICES));
            device_h.unmap_memory(staging.memory);
        }

        // Create device-local vertex buffer (GPU only)
        let raw = RawBuffer::new(
            instance_h,
            physical_device_h,
            device_h,
            size,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        // Copy staging -> device local
        one_time_submit(device_h, device.queue, commands.pool, |cmd_buffer| {
            unsafe {
                device_h.cmd_copy_buffer(
                    cmd_buffer,
                    staging.handle, // src
                    raw.handle,     // dst
                    &[vk::BufferCopy::default().size(size)],
                )
            };
        })?;

        // Staging buffer no longer needed
        unsafe { staging.destroy(device_h) };

        Ok(Self { raw })
    }

    pub fn handle(&self) -> vk::Buffer {
        self.raw.handle
    }

    /// # Safety
    ///
    /// - Must be called before the `ash::Device` that was used to create this
    ///   `VertexBuffer` is destroyed.
    /// - The buffer must not be in use by the GPU (i.e. no command buffer
    ///   currently reading from it is pending execution).
    /// - Must be called at most once. Calling it more than once is undefined
    ///   behaviour as the underlying handles become invalid after the first call.
    pub unsafe fn destroy(&mut self, device: &Device) {
        unsafe { self.raw.destroy(&device.handle) };
    }
}
