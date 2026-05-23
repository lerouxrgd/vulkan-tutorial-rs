use glam::{Mat4, Vec3};

use crate::buffers::{MvpMatrices, UniformBuffers};
use crate::descriptors::{SceneDescriptorPool, SceneDescriptorSets};
use crate::devices::{Device, PhysicalDevice};
use crate::images::{TextureImage, TextureSampler};
use crate::instance::Instance;

#[non_exhaustive]
pub struct GameObject {
    pub position: Vec3,
    pub rotation: Vec3,
    pub scale: Vec3,
    pub uniform_buffers: UniformBuffers<MvpMatrices>,
    pub descriptors: SceneDescriptorSets,
}

impl GameObject {
    pub const COUNT: usize = 3;

    pub fn new(
        instance: &Instance,
        physical_device: &PhysicalDevice,
        device: &Device,
        descriptor_pool: &SceneDescriptorPool,
        texture: &TextureImage,
        sampler: &TextureSampler,
    ) -> anyhow::Result<Self> {
        let max_frames_inflight = descriptor_pool.max_frames_inflight;
        let uniform_buffers =
            UniformBuffers::new(instance, physical_device, device, max_frames_inflight)?;
        let descriptors = descriptor_pool.allocate(device, &uniform_buffers, texture, sampler)?;
        Ok(Self {
            position: Vec3::ZERO,
            rotation: Vec3::ZERO,
            scale: Vec3::ONE,
            uniform_buffers,
            descriptors,
        })
    }

    pub fn update(&mut self, frame_index: usize, delta_time: f32, view: Mat4, proj: Mat4) {
        const ROTATION_SPEED: f32 = 0.5; // radians per second
        self.rotation.y += ROTATION_SPEED * delta_time;

        let ubo = MvpMatrices {
            model: self.model_matrix(),
            view,
            proj,
        };
        self.uniform_buffers.buffers[frame_index].write(ubo);
    }

    fn model_matrix(&self) -> Mat4 {
        Mat4::from_translation(self.position)
            * Mat4::from_rotation_x(self.rotation.x)
            * Mat4::from_rotation_y(self.rotation.y)
            * Mat4::from_rotation_z(self.rotation.z)
            * Mat4::from_scale(self.scale)
    }

    /// # Safety
    ///
    /// - Must be called before the `ash::Device` that was used to create this
    ///   `GameObject` is destroyed.
    /// - Must be called at most once.
    pub unsafe fn destroy(&mut self, device: &Device) {
        unsafe {
            self.uniform_buffers.destroy(device);
        }
    }
}

pub fn setup_game_objects(game_objects: &mut [GameObject]) {
    // Object 1 - Center
    game_objects[0].position = Vec3::ZERO;
    game_objects[0].rotation = Vec3::new(0.0, (-90.0_f32).to_radians(), 0.0);
    game_objects[0].scale = Vec3::ONE;

    // Object 2 - Left
    game_objects[1].position = Vec3::new(-2.0, 0.0, -1.0);
    game_objects[1].rotation = Vec3::new(0.0, (-45.0_f32).to_radians(), 0.0);
    game_objects[1].scale = Vec3::splat(0.75);

    // Object 3 - Right
    game_objects[2].position = Vec3::new(2.0, 0.0, -1.0);
    game_objects[2].rotation = Vec3::new(0.0, 45.0_f32.to_radians(), 0.0);
    game_objects[2].scale = Vec3::splat(0.75);
}
