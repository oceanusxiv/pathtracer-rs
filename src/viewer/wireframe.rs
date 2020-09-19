use super::vertex::VertexPos;
use super::{pipeline::create_render_pipeline, shaders, Instance, Mesh, ViewerScene};
use crate::common::math::cantor_pairing;
use itertools::Itertools;
use std::collections::HashSet;

pub struct WireFrameHandle {
    pub vertex_buffer: wgpu::Buffer,
    pub num_elements: usize,
}

impl WireFrameHandle {
    pub fn from_mesh(device: &wgpu::Device, mesh: &Mesh) -> Self {
        let mut line_set: HashSet<usize> = HashSet::new();
        let mut line_list = vec![];

        for i in (0..mesh.indices.len()).step_by(3) {
            let i0 = mesh.indices[i] as usize;
            let i1 = mesh.indices[i + 1] as usize;
            let i2 = mesh.indices[i + 2] as usize;
            let v0 = mesh.pos[i0];
            let v1 = mesh.pos[i1];
            let v2 = mesh.pos[i2];
            let c01 = cantor_pairing(i0, i1);
            let c12 = cantor_pairing(i1, i2);
            let c20 = cantor_pairing(i2, i0);

            if !line_set.contains(&c01) {
                line_list.push(VertexPos::from(&v0));
                line_list.push(VertexPos::from(&v1));
                line_set.insert(c01);
            }
            if !line_set.contains(&c12) {
                line_list.push(VertexPos::from(&v1));
                line_list.push(VertexPos::from(&v2));
                line_set.insert(c12);
            }
            if !line_set.contains(&c20) {
                line_list.push(VertexPos::from(&v2));
                line_list.push(VertexPos::from(&v0));
                line_set.insert(c20);
            }
        }

        let vertex_buffer = device
            .create_buffer_with_data(bytemuck::cast_slice(&line_list), wgpu::BufferUsage::VERTEX);

        WireFrameHandle {
            vertex_buffer,
            num_elements: line_list.len(),
        }
    }
}

pub struct WireFrameInstancesHandle {
    pub wireframe: WireFrameHandle,
    pub instance_buffer: wgpu::Buffer,
    pub instance_buffer_size: usize,
    pub instances_bind_group: wgpu::BindGroup,
    pub visible_instances: std::ops::Range<u32>,
}

impl WireFrameInstancesHandle {
    pub fn new(
        device: &wgpu::Device,
        instances_bind_group_layout: &wgpu::BindGroupLayout,
        instances: &Vec<na::Projective3<f32>>,
        wireframe: WireFrameHandle,
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

        WireFrameInstancesHandle {
            wireframe,
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

pub struct WireFrameRenderPass {
    render_pipeline: wgpu::RenderPipeline,
    wireframe_instances: Vec<WireFrameInstancesHandle>,
}

impl WireFrameRenderPass {
    pub fn from_scene(
        device: &wgpu::Device,
        compiler: &mut shaderc::Compiler,
        uniform_bind_group_layout: &wgpu::BindGroupLayout,
        scene: &ViewerScene,
    ) -> Self {
        let (vs_module, fs_module) = shaders::flat_instance::compile_shaders(compiler, &device);
        let instances_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                bindings: &[Instance::create_bind_group_layout_entry()],
                label: Some("instances_bind_group_layout"),
            });

        let wireframe_instances = scene
            .meshes
            .iter()
            .map(|mesh| {
                WireFrameInstancesHandle::new(
                    &device,
                    &instances_bind_group_layout,
                    &mesh.instances,
                    WireFrameHandle::from_mesh(&device, &mesh),
                )
            })
            .collect_vec();

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                bind_group_layouts: &[&uniform_bind_group_layout, &instances_bind_group_layout],
            });

        let render_pipeline = create_render_pipeline::<VertexPos>(
            &device,
            render_pipeline_layout,
            &vs_module,
            &fs_module,
            wgpu::PrimitiveTopology::LineList,
            true,
        );

        WireFrameRenderPass {
            wireframe_instances,
            render_pipeline,
        }
    }
}

pub trait DrawWireFrame<'a, 'b>
where
    'b: 'a,
{
    fn draw_wire_frame(&mut self, wire_frame: &'b WireFrameInstancesHandle);
    fn draw_all_wire_frame(&mut self, wire_frame: &'b WireFrameRenderPass);
}

impl<'a, 'b> DrawWireFrame<'a, 'b> for wgpu::RenderPass<'a>
where
    'b: 'a,
{
    fn draw_wire_frame(&mut self, wire_frame: &'b WireFrameInstancesHandle) {
        self.set_bind_group(1, &wire_frame.instances_bind_group, &[]);
        self.set_vertex_buffer(0, &wire_frame.wireframe.vertex_buffer, 0, 0);
        self.draw(
            0..wire_frame.wireframe.num_elements as u32,
            wire_frame.visible_instances.clone(),
        );
    }

    fn draw_all_wire_frame(&mut self, wire_frame: &'b WireFrameRenderPass) {
        self.set_pipeline(&wire_frame.render_pipeline);

        for wire_frame_handle in &wire_frame.wireframe_instances {
            self.draw_wire_frame(&wire_frame_handle);
        }
    }
}
