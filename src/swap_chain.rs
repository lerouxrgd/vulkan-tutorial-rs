use anyhow::ensure;
use ash::{khr, vk};

use crate::surface::Surface;

pub struct SwapChain {
    pub fns: khr::swapchain::Device,
    pub handle: vk::SwapchainKHR,
    pub surface_format: vk::SurfaceFormatKHR,
    pub extent: vk::Extent2D,
    pub images: Vec<vk::Image>,
    pub image_views: Vec<vk::ImageView>,
}

impl SwapChain {
    pub fn new(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        device: &ash::Device,
        surface: &Surface,
        window: &sdl3::video::Window,
    ) -> anyhow::Result<Self> {
        let surface_capabilities = unsafe {
            surface
                .fns
                .get_physical_device_surface_capabilities(physical_device, surface.handle)?
        };
        let extent = Self::choose_extent(&surface_capabilities, window);
        let min_image_count = Self::choose_min_image_count(&surface_capabilities);

        let available_formats = unsafe {
            surface
                .fns
                .get_physical_device_surface_formats(physical_device, surface.handle)?
        };
        let surface_format = Self::choose_surface_format(&available_formats)?;

        let available_present_modes = unsafe {
            surface
                .fns
                .get_physical_device_surface_present_modes(physical_device, surface.handle)?
        };
        let present_mode = Self::choose_present_mode(&available_present_modes)?;

        let swapchain_ci = vk::SwapchainCreateInfoKHR::default()
            .surface(surface.handle)
            .min_image_count(min_image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1) // always 1 except for stereoscopic 3D application
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT) // image will be rendered
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE) // image is owned by one queue family at a time
            .pre_transform(surface_capabilities.current_transform) // no transformation
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE) // ignore alpha channel
            .present_mode(present_mode)
            .clipped(true); // we don’t care about the color of pixels that are obscured

        let fns = khr::swapchain::Device::new(instance, device);
        let handle = unsafe { fns.create_swapchain(&swapchain_ci, None)? };
        let images = unsafe { fns.get_swapchain_images(handle)? };

        let image_views = images
            .iter()
            .map(|&image| {
                let image_view_ci = vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(surface_format.format)
                    .subresource_range(
                        // describes what the image’s purpose is and which part of the image should be accessed
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR) // used as color targets
                            .base_mip_level(0) // without any mipmapping levels
                            .level_count(1)
                            .base_array_layer(0) // without multiple layers
                            .layer_count(1),
                    );
                unsafe { device.create_image_view(&image_view_ci, None) }
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            fns,
            handle,
            surface_format,
            extent,
            images,
            image_views,
        })
    }

    fn choose_min_image_count(capabilities: &vk::SurfaceCapabilitiesKHR) -> u32 {
        let min_image_count = capabilities.min_image_count.max(3);
        // 0 is a special value that means that there is no maximum
        if 0 < capabilities.max_image_count && capabilities.max_image_count < min_image_count {
            capabilities.max_image_count
        } else {
            min_image_count
        }
    }

    fn choose_surface_format(
        available_formats: &[vk::SurfaceFormatKHR],
    ) -> anyhow::Result<vk::SurfaceFormatKHR> {
        ensure!(!available_formats.is_empty());
        let surface_format = available_formats
            .iter()
            .find(|f| {
                f.format == vk::Format::B8G8R8A8_SRGB
                    && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .copied()
            .unwrap_or(available_formats[0]);
        Ok(surface_format)
    }

    fn choose_present_mode(
        available_present_modes: &[vk::PresentModeKHR],
    ) -> anyhow::Result<vk::PresentModeKHR> {
        ensure!(available_present_modes.contains(&vk::PresentModeKHR::FIFO));
        let present_mode = if available_present_modes.contains(&vk::PresentModeKHR::MAILBOX) {
            vk::PresentModeKHR::MAILBOX
        } else {
            vk::PresentModeKHR::FIFO
        };
        Ok(present_mode)
    }

    fn choose_extent(
        capabilities: &vk::SurfaceCapabilitiesKHR,
        window: &sdl3::video::Window,
    ) -> vk::Extent2D {
        if capabilities.current_extent.width != u32::MAX {
            return capabilities.current_extent;
        }
        let (width, height) = window.size_in_pixels();
        vk::Extent2D {
            width: width.clamp(
                capabilities.min_image_extent.width,
                capabilities.max_image_extent.width,
            ),
            height: height.clamp(
                capabilities.min_image_extent.height,
                capabilities.max_image_extent.height,
            ),
        }
    }

    /// # Safety
    ///
    /// - Must be called before the `ash::Device` that was used to create this
    ///   `Swapchain` is destroyed.
    /// - Must be called at most once. Calling it more than once is undefined
    ///   behaviour as the underlying handle becomes invalid after the first call.
    pub unsafe fn destroy(&mut self, device: &ash::Device) {
        unsafe {
            for &image_view in &self.image_views {
                device.destroy_image_view(image_view, None);
            }
            self.fns.destroy_swapchain(self.handle, None)
        };
    }
}
