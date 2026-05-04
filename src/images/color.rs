use ash::vk;

use crate::devices::{Device, PhysicalDevice};
use crate::images::raw::RawImage;
use crate::instance::Instance;

#[non_exhaustive]
pub struct ColorImage {
    raw: RawImage,
    pub view: vk::ImageView,
}

impl ColorImage {
    pub fn new(
        instance: &Instance,
        physical_device: &PhysicalDevice,
        device: &Device,
        width: u32,
        height: u32,
        format: vk::Format,
    ) -> anyhow::Result<Self> {
        let device_h = &device.handle;

        let raw = RawImage::new(
            &instance.handle,
            physical_device.handle,
            device_h,
            width,
            height,
            format,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::TRANSIENT_ATTACHMENT | vk::ImageUsageFlags::COLOR_ATTACHMENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            1,
            physical_device.msaa_samples,
        )?;

        let view_ci = vk::ImageViewCreateInfo::default()
            .image(raw.handle)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            );
        let view = unsafe { device_h.create_image_view(&view_ci, None)? };

        Ok(Self { raw, view })
    }

    pub fn handle(&self) -> vk::Image {
        self.raw.handle
    }

    /// # Safety
    ///
    /// - Must be called before the `ash::Device` that was used to create this
    ///   `ColorImage` is destroyed.
    /// - The image must not be in use by the GPU.
    /// - Must be called at most once. Calling it more than once is undefined
    ///   behaviour as the underlying handles become invalid after the first call.
    pub unsafe fn destroy(&mut self, device: &Device) {
        unsafe {
            device.handle.destroy_image_view(self.view, None);
            self.raw.destroy(&device.handle);
        }
    }
}
