use super::vertex::VertexPosNorm;
use crate::common::{Mesh, World};
use itertools::{zip_eq, Itertools};

pub struct DrawMesh {
    pub index: usize,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub num_elements: usize,
}

impl DrawMesh {
    pub fn from_mesh(device: &wgpu::Device, mesh: &Mesh) -> Self {
        let vertices = zip_eq(&mesh.pos, &mesh.normal)
            .map_into::<VertexPosNorm>()
            .collect_vec();

        let vertex_buffer = device.create_buffer_with_data(
            bytemuck::cast_slice(&vertices[..]),
            wgpu::BufferUsage::VERTEX,
        );

        let index_buffer = device.create_buffer_with_data(
            bytemuck::cast_slice(&mesh.indices[..]),
            wgpu::BufferUsage::INDEX,
        );

        DrawMesh {
            index: mesh.index,
            vertex_buffer,
            index_buffer,
            num_elements: mesh.indices.len(),
        }
    }
}

pub struct DrawMeshInstances {
    pub mesh: DrawMesh,
    pub instance_buffer: wgpu::Buffer,
    pub instance_buffer_size: usize,
    pub instances_bind_group: wgpu::BindGroup,
    pub visible_instances: std::ops::Range<u32>,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Instance {
    model: glm::Mat4,
}

unsafe impl bytemuck::Zeroable for Instance {}

unsafe impl bytemuck::Pod for Instance {}

impl Instance {
    pub fn create_bind_group_layout_entry() -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStage::VERTEX,
            ty: wgpu::BindingType::StorageBuffer {
                // We don't plan on changing the size of this buffer
                dynamic: false,
                // The shader is not allowed to modify it's contents
                readonly: true,
            },
        }
    }
}

impl DrawMeshInstances {
    pub fn from_world(
        device: &wgpu::Device,
        instances_bind_group_layout: &wgpu::BindGroupLayout,
        world: &World,
        mesh: DrawMesh,
    ) -> Self {
        let instance_data = world
            .objects
            .iter()
            .filter(|obj| obj.mesh.index == mesh.index)
            .map(|obj| Instance {
                model: obj.obj_to_world.to_homogeneous(),
            })
            .collect_vec();

        let instance_buffer_size = instance_data.len() * std::mem::size_of::<glm::Mat4>();
        let instance_buffer = device.create_buffer_with_data(
            bytemuck::cast_slice(&instance_data[..]),
            wgpu::BufferUsage::STORAGE_READ,
        );

        let instances_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &instances_bind_group_layout,
            bindings: &[wgpu::Binding {
                binding: 0,
                resource: wgpu::BindingResource::Buffer {
                    buffer: &instance_buffer,
                    range: 0..instance_buffer_size as wgpu::BufferAddress,
                },
            }],
            label: Some("instances_bind_group"),
        });

        DrawMeshInstances {
            mesh,
            instance_buffer,
            instance_buffer_size,
            instances_bind_group,
            visible_instances: std::ops::Range {
                start: 0,
                end: instance_data.len() as u32,
            },
        }
    }
}
