use super::vertex::VertexPosNorm;
use super::{pipeline::create_render_pipeline, shaders, Instance, Mesh, ViewerScene};
use itertools::{zip_eq, Itertools};

pub struct MeshHandle {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub num_elements: usize,
}

impl MeshHandle {
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

        MeshHandle {
            vertex_buffer,
            index_buffer,
            num_elements: mesh.indices.len(),
        }
    }
}

pub struct MeshInstancesHandle {
    pub mesh: MeshHandle,
    pub instance_buffer: wgpu::Buffer,
    pub instance_buffer_size: usize,
    pub instances_bind_group: wgpu::BindGroup,
    pub visible_instances: std::ops::Range<u32>,
}

impl MeshInstancesHandle {
    pub fn new(
        device: &wgpu::Device,
        instances_bind_group_layout: &wgpu::BindGroupLayout,
        instances: &Vec<na::Projective3<f32>>,
        mesh: MeshHandle,
    ) -> Self {
        let instance_data = instances
            .iter()
            .map(|trans| Instance {
                model: trans.to_homogeneous(),
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

        MeshInstancesHandle {
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

pub struct MeshRenderPass {
    render_pipeline: wgpu::RenderPipeline,
    draw_mesh_instances: Vec<MeshInstancesHandle>,
}

impl MeshRenderPass {
    pub fn from_scene(
        device: &wgpu::Device,
        mut compiler: &mut shaderc::Compiler,
        uniform_bind_group_layout: &wgpu::BindGroupLayout,
        scene: &ViewerScene,
    ) -> Self {
        let (vs_module, fs_module) = shaders::phong::compile_shaders(&mut compiler, &device);

        let instances_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                bindings: &[Instance::create_bind_group_layout_entry()],
                label: Some("instances_bind_group_layout"),
            });

        let draw_mesh_instances = scene
            .meshes
            .iter()
            .map(|mesh| {
                MeshInstancesHandle::new(
                    &device,
                    &instances_bind_group_layout,
                    &mesh.instances,
                    MeshHandle::from_mesh(&device, &mesh),
                )
            })
            .collect_vec();

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                bind_group_layouts: &[&uniform_bind_group_layout, &instances_bind_group_layout],
            });

        let render_pipeline = create_render_pipeline::<VertexPosNorm>(
            &device,
            render_pipeline_layout,
            &vs_module,
            &fs_module,
            wgpu::PrimitiveTopology::TriangleList,
            true,
        );

        MeshRenderPass {
            draw_mesh_instances,
            render_pipeline,
        }
    }
}

pub trait DrawMesh<'a, 'b>
where
    'b: 'a,
{
    fn draw_mesh_instances(&mut self, mesh: &'b MeshInstancesHandle);
    fn draw_all_mesh(&mut self, mesh: &'b MeshRenderPass);
}

impl<'a, 'b> DrawMesh<'a, 'b> for wgpu::RenderPass<'a>
where
    'b: 'a,
{
    fn draw_mesh_instances(&mut self, mesh_instances: &'b MeshInstancesHandle) {
        self.set_bind_group(1, &mesh_instances.instances_bind_group, &[]);
        self.set_vertex_buffer(0, &mesh_instances.mesh.vertex_buffer, 0, 0);
        self.set_index_buffer(&mesh_instances.mesh.index_buffer, 0, 0);
        self.draw_indexed(
            0..mesh_instances.mesh.num_elements as u32,
            0,
            mesh_instances.visible_instances.clone(),
        );
    }

    fn draw_all_mesh(&mut self, meshes: &'b MeshRenderPass) {
        self.set_pipeline(&meshes.render_pipeline);

        for mesh_instance in &meshes.draw_mesh_instances {
            self.draw_mesh_instances(&mesh_instance);
        }
    }
}
