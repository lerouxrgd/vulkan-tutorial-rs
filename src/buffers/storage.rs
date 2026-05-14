use std::f32::consts::TAU;
use std::mem;
use std::slice;

use ash::vk;
use bytemuck::{Pod, Zeroable};
use glam::{Vec2, Vec4};
use rand::RngExt;

use crate::buffers::raw::RawBuffer;
use crate::commands::{ParticlesCommands, one_time_submit};
use crate::devices::{Device, PhysicalDevice};
use crate::instance::Instance;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Particle {
    pub position: Vec2,
    pub velocity: Vec2,
    pub color: Vec4,
}

impl Particle {
    pub const COUNT: usize = 8192;

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
                .format(vk::Format::R32G32_SFLOAT)
                .offset(mem::offset_of!(Self, position) as u32),
            vk::VertexInputAttributeDescription::default()
                .location(1)
                .binding(0)
                .format(vk::Format::R32G32B32A32_SFLOAT)
                .offset(mem::offset_of!(Self, color) as u32),
        ]
    }

    fn make_particles(width: u32, height: u32) -> Vec<Self> {
        let mut rng = rand::rng();
        (0..Self::COUNT)
            .map(|_| {
                let r = 0.25 * rng.random::<f32>().sqrt();
                let theta = rng.random::<f32>() * TAU;
                let x = r * theta.cos() * height as f32 / width as f32;
                let y = r * theta.sin();
                let position = Vec2::new(x, y);
                let velocity = position.normalize() * 0.00025;
                let color = Vec4::new(rng.random(), rng.random(), rng.random(), 1.0);
                Self {
                    position,
                    velocity,
                    color,
                }
            })
            .collect()
    }
}

#[non_exhaustive]
pub struct StorageBuffer {
    raw: RawBuffer,
}

impl StorageBuffer {
    pub fn handle(&self) -> vk::Buffer {
        self.raw.handle
    }

    /// # Safety
    ///
    /// - Must be called before the `ash::Device` that was used to create this
    ///   `StorageBuffer` is destroyed.
    /// - The buffer must not be in use by the GPU.
    /// - Must be called at most once.
    pub unsafe fn destroy(&mut self, device: &Device) {
        unsafe { self.raw.destroy(&device.handle) };
    }
}

pub struct StorageBuffers {
    pub buffers: Vec<StorageBuffer>,
}

impl StorageBuffers {
    pub fn new(
        instance: &Instance,
        physical_device: &PhysicalDevice,
        device: &Device,
        commands: &ParticlesCommands,
        max_frames_in_flight: usize,
        width: u32,
        height: u32,
    ) -> anyhow::Result<Self> {
        let instance_h = &instance.handle;
        let physical_device_h = physical_device.handle;
        let device_h = &device.handle;

        let particles = Particle::make_particles(width, height);
        let size = mem::size_of_val(particles.as_slice()) as vk::DeviceSize;

        // Staging buffer
        let mut staging = RawBuffer::new(
            instance_h,
            physical_device_h,
            device_h,
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        unsafe {
            let data = device_h.map_memory(staging.memory, 0, size, vk::MemoryMapFlags::empty())?;
            let slice = slice::from_raw_parts_mut(data as *mut u8, size as usize);
            slice.copy_from_slice(bytemuck::cast_slice(&particles));
            device_h.unmap_memory(staging.memory);
        }

        // One storage buffer per frame, each initialized from staging
        let buffers = (0..max_frames_in_flight)
            .map(|_| {
                let raw = RawBuffer::new(
                    instance_h,
                    physical_device_h,
                    device_h,
                    size,
                    vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::VERTEX_BUFFER
                        | vk::BufferUsageFlags::TRANSFER_DST,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                )?;

                one_time_submit(device_h, device.queue, commands.pool, |cmd| unsafe {
                    device_h.cmd_copy_buffer(
                        cmd,
                        staging.handle,
                        raw.handle,
                        &[vk::BufferCopy::default().size(size)],
                    );
                })?;

                Ok(StorageBuffer { raw })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        unsafe { staging.destroy(device_h) };

        Ok(Self { buffers })
    }

    /// # Safety
    /// Same conditions as `StorageBuffer::destroy`, applied to all buffers.
    pub unsafe fn destroy(&mut self, device: &Device) {
        for buf in &mut self.buffers {
            unsafe { buf.destroy(device) };
        }
    }
}
