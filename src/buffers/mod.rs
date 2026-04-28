mod index;
mod raw;
mod uniform;
mod vertex;

pub use self::index::IndexBuffer;
pub use self::raw::{RawBuffer, find_memory_type};
pub use self::uniform::{UniformBufferObject, UniformBuffers};
pub use self::vertex::{Vertex, VertexBuffer};

pub struct Model {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

impl Model {
    pub fn from_obj<P>(model_file: P) -> anyhow::Result<Self>
    where
        P: AsRef<std::path::Path>,
    {
        let (models, _materials) = tobj::load_obj(model_file.as_ref(), &tobj::GPU_LOAD_OPTIONS)?;

        let mut vertices: Vec<Vertex> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();
        let mut unique_vertices: std::collections::HashMap<u32, u32> = Default::default();

        for model in &models {
            let mesh = &model.mesh;
            for &i in &mesh.indices {
                let index = *unique_vertices.entry(i).or_insert_with(|| {
                    let idx = vertices.len() as u32;
                    let vi = i as usize;
                    vertices.push(Vertex {
                        pos: glam::Vec3::new(
                            mesh.positions[vi * 3],
                            mesh.positions[vi * 3 + 1],
                            mesh.positions[vi * 3 + 2],
                        ),
                        color: glam::Vec3::ONE,
                        tex_coord: glam::Vec2::new(
                            mesh.texcoords[vi * 2],
                            1.0 - mesh.texcoords[vi * 2 + 1],
                        ),
                    });
                    idx
                });
                indices.push(index);
            }
        }
        log::debug!(
            "Loaded {} vertices and {} indices from {}",
            vertices.len(),
            indices.len(),
            model_file.as_ref().display()
        );

        Ok(Self { vertices, indices })
    }
}
