mod index;
mod raw;
mod storage;
mod uniform;
mod vertex;

pub use self::index::IndexBuffer;
pub use self::raw::{RawBuffer, find_memory_type};
pub use self::storage::{Particle, StorageBuffers};
pub use self::uniform::{DeltaTime, MvpMatrices, UniformBuffers};
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
                            mesh.positions[vi * 3 + 2],  // Z becomes Y
                            -mesh.positions[vi * 3 + 1], // Y becomes -Z
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

    pub fn from_gltf<P>(model_file: P) -> anyhow::Result<Self>
    where
        P: AsRef<std::path::Path>,
    {
        let (document, buffers, _images) = gltf::import(model_file.as_ref())?;

        let mut vertices: Vec<Vertex> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();

        for mesh in document.meshes() {
            for primitive in mesh.primitives() {
                let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

                let positions: Vec<[f32; 3]> = reader
                    .read_positions()
                    .ok_or_else(|| anyhow::anyhow!("primitive has no positions"))?
                    .collect();

                let tex_coords: Vec<[f32; 2]> = reader
                    .read_tex_coords(0)
                    .map(|t| t.into_f32().collect())
                    .unwrap_or_else(|| vec![[0.0, 0.0]; positions.len()]);

                let vertex_offset = vertices.len() as u32;

                for (pos, tex_coord) in positions.iter().zip(tex_coords.iter()) {
                    vertices.push(Vertex {
                        pos: glam::Vec3::new(pos[0], pos[1], pos[2]),
                        color: glam::Vec3::ONE,
                        tex_coord: glam::Vec2::new(tex_coord[0], tex_coord[1]), // no Y flip for glTF
                    });
                }

                if let Some(iter) = reader.read_indices() {
                    for i in iter.into_u32() {
                        indices.push(vertex_offset + i);
                    }
                } else {
                    // No index buffer — generate sequential indices
                    for i in 0..positions.len() as u32 {
                        indices.push(vertex_offset + i);
                    }
                }
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
