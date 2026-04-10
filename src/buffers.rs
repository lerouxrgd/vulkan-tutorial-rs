use std::{mem, slice};

use anyhow::anyhow;
use ash::vk;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec2, Vec3};

use crate::commands::Commands;
use crate::devices::{Device, PhysicalDevice};
use crate::instance::Instance;
use crate::swap_chain::SwapChain;

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
    pub handle: vk::Buffer,
    pub memory: vk::DeviceMemory,
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

        let size = (mem::size_of::<Vertex>() * Self::VERTICES.len()) as vk::DeviceSize;

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
        let RawBuffer { handle, memory } = RawBuffer::new(
            instance_h,
            physical_device_h,
            device_h,
            size,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        // Copy staging -> device local
        copy_buffer(
            device_h,
            device.queue,
            commands.pool,
            staging.handle,
            handle,
            size,
        )?;

        // Staging buffer no longer needed
        unsafe { staging.destroy(device) };

        Ok(Self { handle, memory })
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
        unsafe {
            device.handle.destroy_buffer(self.handle, None);
            device.handle.free_memory(self.memory, None);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////

#[non_exhaustive]
pub struct IndexBuffer {
    pub handle: vk::Buffer,
    pub memory: vk::DeviceMemory,
}

impl IndexBuffer {
    pub const INDICES: &[u16] = &[0, 1, 2, 2, 3, 0];

    pub fn new(
        instance: &Instance,
        physical_device: &PhysicalDevice,
        device: &Device,
        commands: &Commands,
    ) -> anyhow::Result<Self> {
        let instance_h = &instance.handle;
        let physical_device_h = physical_device.handle;
        let device_h = &device.handle;

        let size = (mem::size_of::<u16>() * Self::INDICES.len()) as vk::DeviceSize;

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
            slice.copy_from_slice(bytemuck::cast_slice(Self::INDICES));
            device_h.unmap_memory(staging.memory);
        }

        let RawBuffer { handle, memory } = RawBuffer::new(
            instance_h,
            physical_device_h,
            device_h,
            size,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        copy_buffer(
            device_h,
            device.queue,
            commands.pool,
            staging.handle,
            handle,
            size,
        )?;

        unsafe { staging.destroy(device) };

        Ok(Self { handle, memory })
    }

    pub fn len(&self) -> u32 {
        Self::INDICES.len() as u32
    }

    /// # Safety
    ///
    /// - Must be called before the `ash::Device` that was used to create this
    ///   `IndexBuffer` is destroyed.
    /// - The buffer must not be in use by the GPU (i.e. no command buffer
    ///   currently reading from it is pending execution).
    /// - Must be called at most once. Calling it more than once is undefined
    ///   behaviour as the underlying handles become invalid after the first call.
    pub unsafe fn destroy(&mut self, device: &Device) {
        unsafe {
            device.handle.destroy_buffer(self.handle, None);
            device.handle.free_memory(self.memory, None);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct UniformBufferObject {
    pub model: Mat4,
    pub view: Mat4,
    pub proj: Mat4,
}

pub struct UniformBuffer {
    raw: RawBuffer,
    pub mapped: *mut UniformBufferObject,
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
            unsafe { buffer.raw.destroy(device) };
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////

struct RawBuffer {
    handle: vk::Buffer,
    memory: vk::DeviceMemory,
}

impl RawBuffer {
    fn new(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        device: &ash::Device,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        properties: vk::MemoryPropertyFlags,
    ) -> anyhow::Result<Self> {
        let buffer_ci = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE); // owned by one queue only
        let handle = unsafe { device.create_buffer(&buffer_ci, None)? };

        let mem_requirements = unsafe { device.get_buffer_memory_requirements(handle) };
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

        unsafe { device.bind_buffer_memory(handle, memory, 0)? };

        Ok(RawBuffer { handle, memory })
    }

    /// # Safety
    ///
    /// - Must be called before the `ash::Device` that was used to create this
    ///   `RawBuffer` is destroyed.
    /// - The buffer must not be in use by the GPU (i.e. no command buffer
    ///   currently reading from it is pending execution).
    /// - Must be called at most once. Calling it more than once is undefined
    ///   behaviour as the underlying handles become invalid after the first call.
    unsafe fn destroy(&mut self, device: &Device) {
        unsafe {
            device.handle.destroy_buffer(self.handle, None);
            device.handle.free_memory(self.memory, None);
        }
    }
}

fn find_memory_type(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    type_filter: u32,
    properties: vk::MemoryPropertyFlags,
) -> anyhow::Result<u32> {
    let mem_properties = unsafe { instance.get_physical_device_memory_properties(physical_device) };
    (0..mem_properties.memory_type_count)
        .find(|&i| {
            type_filter & (1 << i) != 0
                && mem_properties.memory_types[i as usize]
                    .property_flags
                    .contains(properties)
        })
        .ok_or_else(|| anyhow!("Failed to find suitable memory type"))
}

fn copy_buffer(
    device: &ash::Device,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    src: vk::Buffer,
    dst: vk::Buffer,
    size: vk::DeviceSize,
) -> anyhow::Result<()> {
    // Allocate a short-lived command buffer
    let alloc_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let cmd_buffer = unsafe { device.allocate_command_buffers(&alloc_info)?[0] };

    // Record the copy
    let begin_info =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    unsafe {
        device.begin_command_buffer(cmd_buffer, &begin_info)?;
        device.cmd_copy_buffer(
            cmd_buffer,
            src,
            dst,
            &[vk::BufferCopy::default().size(size)],
        );
        device.end_command_buffer(cmd_buffer)?;
    }

    // Submit and wait
    let submit_info = vk::SubmitInfo::default().command_buffers(slice::from_ref(&cmd_buffer));
    unsafe {
        device.queue_submit(queue, slice::from_ref(&submit_info), vk::Fence::null())?;
        device.queue_wait_idle(queue)?;
        device.free_command_buffers(command_pool, &[cmd_buffer]);
    }

    Ok(())
}
