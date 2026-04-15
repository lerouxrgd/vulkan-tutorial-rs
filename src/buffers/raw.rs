use anyhow::anyhow;
use ash::vk;

#[non_exhaustive]
pub struct RawBuffer {
    pub handle: vk::Buffer,
    pub memory: vk::DeviceMemory,
}

impl RawBuffer {
    pub fn new(
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
    pub unsafe fn destroy(&mut self, device: &ash::Device) {
        unsafe {
            device.destroy_buffer(self.handle, None);
            device.free_memory(self.memory, None);
        }
    }
}

pub fn find_memory_type(
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
