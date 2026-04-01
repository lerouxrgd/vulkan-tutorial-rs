use std::{mem, slice};

use anyhow::anyhow;
use ash::vk;
use bytemuck::{Pod, Zeroable};
use glam::{Vec2, Vec3};

use crate::devices::{Device, PhysicalDevice};
use crate::instance::Instance;

pub const VERTICES: &[Vertex] = &[
    Vertex {
        pos: Vec2::new(0.0, -0.5),
        color: Vec3::new(1.0, 0.0, 0.0),
    },
    Vertex {
        pos: Vec2::new(0.5, 0.5),
        color: Vec3::new(0.0, 1.0, 0.0),
    },
    Vertex {
        pos: Vec2::new(-0.5, 0.5),
        color: Vec3::new(0.0, 0.0, 1.0),
    },
];

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

// TODO: use crate internal types here
impl VertexBuffer {
    pub fn new(
        instance: &Instance,
        physical_device: &PhysicalDevice,
        device: &Device,
    ) -> anyhow::Result<Self> {
        let instance = &instance.handle;
        let physical_device = physical_device.handle;
        let device = &device.handle;

        let size = (mem::size_of::<Vertex>() * VERTICES.len()) as vk::DeviceSize;

        let buffer_ci = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::VERTEX_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE); // owned by one queue only
        let handle = unsafe { device.create_buffer(&buffer_ci, None)? };

        let mem_requirements = unsafe { device.get_buffer_memory_requirements(handle) };
        let memory_type_index = find_memory_type(
            instance,
            physical_device,
            mem_requirements.memory_type_bits,
            // The below flags mean: GPU memory is directly mappable from the CPU and
            // writes are immediately visible without explicit flushing
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_requirements.size)
            .memory_type_index(memory_type_index);
        let memory = unsafe { device.allocate_memory(&alloc_info, None)? };

        unsafe {
            device.bind_buffer_memory(handle, memory, 0)?;
            let data = device.map_memory(memory, 0, size, vk::MemoryMapFlags::empty())?;
            let slice = slice::from_raw_parts_mut(data as *mut u8, size as usize);
            slice.copy_from_slice(bytemuck::cast_slice(VERTICES));
            device.unmap_memory(memory);
        }

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
