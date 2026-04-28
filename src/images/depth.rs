use anyhow::anyhow;
use ash::vk;

use crate::devices::{Device, PhysicalDevice};
use crate::images::raw::RawImage;
use crate::instance::Instance;

#[non_exhaustive]
pub struct DepthImage {
    raw: RawImage,
    pub view: vk::ImageView,
    pub format: vk::Format,
}

impl DepthImage {
    pub fn new(
        instance: &Instance,
        physical_device: &PhysicalDevice,
        device: &Device,
        width: u32,
        height: u32,
    ) -> anyhow::Result<Self> {
        let device_h = &device.handle;

        let format = find_supported_format(
            &instance.handle,
            physical_device.handle,
            &[
                vk::Format::D32_SFLOAT,
                vk::Format::D32_SFLOAT_S8_UINT,
                vk::Format::D24_UNORM_S8_UINT,
            ],
            vk::ImageTiling::OPTIMAL,
            vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT,
        )?;
        log::debug!("Selected depth format: {format:?}");

        let raw = RawImage::new(
            &instance.handle,
            physical_device.handle,
            device_h,
            width,
            height,
            format,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        let view_ci = vk::ImageViewCreateInfo::default()
            .image(raw.handle)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::DEPTH)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            );
        let view = unsafe { device_h.create_image_view(&view_ci, None)? };

        Ok(Self { raw, view, format })
    }

    pub fn handle(&self) -> vk::Image {
        self.raw.handle
    }

    // pub fn has_stencil_component(&self) -> bool {
    //     matches!(
    //         self.format,
    //         vk::Format::D32_SFLOAT_S8_UINT | vk::Format::D24_UNORM_S8_UINT
    //     )
    // }

    /// # Safety
    ///
    /// - Must be called before the `ash::Device` that was used to create this
    ///   `DepthImage` is destroyed.
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

fn find_supported_format(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    candidates: &[vk::Format],
    tiling: vk::ImageTiling,
    features: vk::FormatFeatureFlags,
) -> anyhow::Result<vk::Format> {
    candidates
        .iter()
        .copied()
        .find(|&format| {
            let props =
                unsafe { instance.get_physical_device_format_properties(physical_device, format) };
            match tiling {
                vk::ImageTiling::LINEAR => props.linear_tiling_features.contains(features),
                vk::ImageTiling::OPTIMAL => props.optimal_tiling_features.contains(features),
                _ => false,
            }
        })
        .ok_or_else(|| anyhow!("failed to find supported format"))
}
