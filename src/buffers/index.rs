use std::{mem, slice};

use ash::vk;

use crate::buffers::raw::RawBuffer;
use crate::commands::{Commands, one_time_submit};
use crate::devices::{Device, PhysicalDevice};
use crate::instance::Instance;

pub struct IndexBuffer {
    raw: RawBuffer,
    pub length: u32,
}

impl IndexBuffer {
    pub fn new(
        instance: &Instance,
        physical_device: &PhysicalDevice,
        device: &Device,
        commands: &Commands,
        indices: &[u32],
    ) -> anyhow::Result<Self> {
        let instance_h = &instance.handle;
        let physical_device_h = physical_device.handle;
        let device_h = &device.handle;

        let size = mem::size_of_val(indices) as vk::DeviceSize;

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
            slice.copy_from_slice(bytemuck::cast_slice(indices));
            device_h.unmap_memory(staging.memory);
        }

        let raw = RawBuffer::new(
            instance_h,
            physical_device_h,
            device_h,
            size,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

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

        unsafe { staging.destroy(device_h) };

        Ok(Self {
            raw,
            length: indices.len() as u32,
        })
    }

    pub fn handle(&self) -> vk::Buffer {
        self.raw.handle
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
        unsafe { self.raw.destroy(&device.handle) };
    }
}
