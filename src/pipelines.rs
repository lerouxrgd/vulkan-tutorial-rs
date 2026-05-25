use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::slice;

use anyhow::{anyhow, ensure};
use ash::vk;

use crate::buffers::{Particle, Vertex};
use crate::descriptors::{ParticlesDescriptors, SceneDescriptorPool};
use crate::devices::{Device, PhysicalDevice};
use crate::swap_chain::SwapChain;
use crate::worker::ParticleGroup;

#[non_exhaustive]
pub struct ShaderModule {
    pub handle: vk::ShaderModule,
}

impl ShaderModule {
    pub fn from_spv_file<P: AsRef<Path>>(device: &Device, path: P) -> anyhow::Result<Self> {
        let mut file = File::open(path)?;

        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;

        ensure!(
            bytes.len() % 4 == 0,
            "SPIR-V binary size must be a multiple of 4 bytes"
        );
        let code: &[u32] = bytemuck::try_cast_slice(&bytes)
            .map_err(|e| anyhow!("SPIR-V buffer is not valid as &[u32]: {e}"))?;

        let shader_module_ci = vk::ShaderModuleCreateInfo::default().code(code);
        let handle = unsafe {
            device
                .handle
                .create_shader_module(&shader_module_ci, None)?
        };

        Ok(Self { handle })
    }

    /// # Safety
    ///
    /// - Must be called before the `ash::Device` that was used to create this
    ///   `ShaderModule` is destroyed.
    /// - Must be called at most once. Calling it more than once is undefined
    ///   behaviour as the underlying handle becomes invalid after the first call.
    pub unsafe fn destroy(&mut self, device: &Device) {
        unsafe { device.handle.destroy_shader_module(self.handle, None) };
    }
}

////////////////////////////////////////////////////////////////////////////////////////

#[non_exhaustive]
pub struct ScenePipeline {
    pub handle: vk::Pipeline,
    pub layout: vk::PipelineLayout,
}

impl ScenePipeline {
    pub fn new<P: AsRef<Path>>(
        physical_device: &PhysicalDevice,
        device: &Device,
        swap_chain: &SwapChain,
        descriptor_pool: &SceneDescriptorPool,
        spv_path: P,
    ) -> anyhow::Result<Self> {
        let mut shader_module = ShaderModule::from_spv_file(device, spv_path)?;

        let shader_stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(shader_module.handle)
                .name(c"vertMain"),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(shader_module.handle)
                .name(c"fragMain"),
        ];

        let vertex_attribute_description = Vertex::attribute_descriptions();
        let vertex_binding_description = Vertex::binding_description();
        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(slice::from_ref(&vertex_binding_description))
            .vertex_attribute_descriptions(&vertex_attribute_description);

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .depth_bias_enable(false)
            .line_width(1.0);

        let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(physical_device.msaa_samples)
            .sample_shading_enable(false) // disable sample shading in the pipeline
            .min_sample_shading(0.2); // min fraction for sample shading; closer to one is smoother

        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL)
            .depth_bounds_test_enable(false)
            .stencil_test_enable(false);

        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(false)
            .color_write_mask(vk::ColorComponentFlags::RGBA);

        let color_blending = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .logic_op(vk::LogicOp::COPY)
            .attachments(slice::from_ref(&color_blend_attachment));

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        let pipeline_layout_ci = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(slice::from_ref(&descriptor_pool.desc_set_layout));
        let layout = unsafe {
            device
                .handle
                .create_pipeline_layout(&pipeline_layout_ci, None)?
        };

        let mut pipeline_rendering_ci = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(slice::from_ref(&swap_chain.surface_format.format))
            .depth_attachment_format(swap_chain.depth_image.format);

        let pipeline_ci = vk::GraphicsPipelineCreateInfo::default()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input_state)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .depth_stencil_state(&depth_stencil)
            .color_blend_state(&color_blending)
            .dynamic_state(&dynamic_state)
            .layout(layout)
            .push_next(&mut pipeline_rendering_ci);

        let handle = unsafe {
            device
                .handle
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    slice::from_ref(&pipeline_ci),
                    None,
                )
                .map_err(|(_, e)| e)?[0]
        };

        // Shader module is no longer needed once the pipeline is compiled
        unsafe { shader_module.destroy(device) };

        Ok(Self { handle, layout })
    }

    /// # Safety
    ///
    /// - Must be called before the `ash::Device` that was used to create this
    ///   `ScenePipeline` is destroyed.
    /// - Must be called at most once. Calling it more than once is undefined
    ///   behaviour as the underlying handle becomes invalid after the first call.
    pub unsafe fn destroy(&mut self, device: &Device) {
        unsafe {
            device.handle.destroy_pipeline(self.handle, None);
            device.handle.destroy_pipeline_layout(self.layout, None);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////

#[non_exhaustive]
pub struct ParticlesPipeline {
    pub handle: vk::Pipeline,
    pub layout: vk::PipelineLayout,
}

impl ParticlesPipeline {
    pub fn new<P: AsRef<Path>>(
        device: &Device,
        swap_chain: &SwapChain,
        spv_path: P,
    ) -> anyhow::Result<Self> {
        let mut shader_module = ShaderModule::from_spv_file(device, spv_path)?;

        let shader_stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(shader_module.handle)
                .name(c"vertMain"),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(shader_module.handle)
                .name(c"fragMain"),
        ];

        let binding_description = Particle::binding_description();
        let attribute_descriptions = Particle::attribute_descriptions();

        let vertex_input_ci = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(slice::from_ref(&binding_description))
            .vertex_attribute_descriptions(&attribute_descriptions);

        let input_assembly_ci = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::POINT_LIST)
            .primitive_restart_enable(false);

        let viewport_state_ci = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        let rasterizer_ci = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .depth_bias_enable(false)
            .line_width(1.0);

        let multisampling_ci = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1)
            .sample_shading_enable(false);

        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
            .alpha_blend_op(vk::BlendOp::ADD)
            .color_write_mask(vk::ColorComponentFlags::RGBA);

        let color_blending_ci = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .logic_op(vk::LogicOp::COPY)
            .attachments(slice::from_ref(&color_blend_attachment));

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state_ci =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        let layout = unsafe {
            device
                .handle
                .create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default(), None)?
        };

        let mut pipeline_rendering_ci = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(slice::from_ref(&swap_chain.surface_format.format));

        let pipeline_ci = vk::GraphicsPipelineCreateInfo::default()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input_ci)
            .input_assembly_state(&input_assembly_ci)
            .viewport_state(&viewport_state_ci)
            .rasterization_state(&rasterizer_ci)
            .multisample_state(&multisampling_ci)
            .color_blend_state(&color_blending_ci)
            .dynamic_state(&dynamic_state_ci)
            .layout(layout)
            .push_next(&mut pipeline_rendering_ci);

        let handle = unsafe {
            device
                .handle
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    slice::from_ref(&pipeline_ci),
                    None,
                )
                .map_err(|(_, e)| e)?[0]
        };

        unsafe { shader_module.destroy(device) };

        Ok(Self { handle, layout })
    }

    /// # Safety
    ///
    /// - Must be called before the `ash::Device` that was used to create this
    ///   `ParticlesPipeline` is destroyed.
    /// - Must be called at most once.
    pub unsafe fn destroy(&mut self, device: &Device) {
        unsafe {
            device.handle.destroy_pipeline(self.handle, None);
            device.handle.destroy_pipeline_layout(self.layout, None);
        }
    }
}

#[non_exhaustive]
pub struct ParticlesComputePipeline {
    pub handle: vk::Pipeline,
    pub layout: vk::PipelineLayout,
}

impl ParticlesComputePipeline {
    pub fn new<P: AsRef<Path>>(
        device: &Device,
        descriptors: &ParticlesDescriptors,
        spv_path: P,
    ) -> anyhow::Result<Self> {
        let mut shader_module = ShaderModule::from_spv_file(device, spv_path)?;

        let compute_shader_stage_ci = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module.handle)
            .name(c"compMain");

        let layout = {
            let layout_ci = vk::PipelineLayoutCreateInfo::default()
                .set_layouts(slice::from_ref(&descriptors.desc_set_layout));
            unsafe { device.handle.create_pipeline_layout(&layout_ci, None)? }
        };

        let pipeline_ci = vk::ComputePipelineCreateInfo::default()
            .stage(compute_shader_stage_ci)
            .layout(layout);
        let pipeline = unsafe {
            device
                .handle
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    slice::from_ref(&pipeline_ci),
                    None,
                )
                .map_err(|(_, e)| e)?[0]
        };

        unsafe { shader_module.destroy(device) };

        Ok(Self {
            handle: pipeline,
            layout,
        })
    }

    /// # Safety
    ///
    /// - Must be called before the `ash::Device` that was used to create this
    ///   `ParticlesComputePipeline` is destroyed.
    /// - Must be called at most once. Calling it more than once is undefined
    ///   behaviour as the underlying handle becomes invalid after the first call.
    pub unsafe fn destroy(&mut self, device: &Device) {
        unsafe {
            device.handle.destroy_pipeline(self.handle, None);
            device.handle.destroy_pipeline_layout(self.layout, None);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////

#[non_exhaustive]
pub struct ParticlesComputeMtPipeline {
    pub handle: vk::Pipeline,
    pub layout: vk::PipelineLayout,
}

impl ParticlesComputeMtPipeline {
    pub fn new<P: AsRef<Path>>(
        device: &Device,
        descriptors: &ParticlesDescriptors,
        spv_path: P,
    ) -> anyhow::Result<Self> {
        let mut shader_module = ShaderModule::from_spv_file(device, spv_path)?;

        let compute_shader_stage_ci = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module.handle)
            .name(c"compMain");

        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<ParticleGroup>() as u32);

        let layout = {
            let layout_ci = vk::PipelineLayoutCreateInfo::default()
                .set_layouts(slice::from_ref(&descriptors.desc_set_layout))
                .push_constant_ranges(slice::from_ref(&push_constant_range));
            unsafe { device.handle.create_pipeline_layout(&layout_ci, None)? }
        };

        let pipeline_ci = vk::ComputePipelineCreateInfo::default()
            .stage(compute_shader_stage_ci)
            .layout(layout);

        let handle = unsafe {
            device
                .handle
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    slice::from_ref(&pipeline_ci),
                    None,
                )
                .map_err(|(_, e)| e)?[0]
        };

        unsafe { shader_module.destroy(device) };

        Ok(Self { handle, layout })
    }

    /// # Safety
    ///
    /// - Must be called before the `ash::Device` that was used to create this
    ///   `ParticlesComputeMtPipeline` is destroyed.
    /// - Must be called at most once.
    pub unsafe fn destroy(&mut self, device: &Device) {
        unsafe {
            device.handle.destroy_pipeline(self.handle, None);
            device.handle.destroy_pipeline_layout(self.layout, None);
        }
    }
}
