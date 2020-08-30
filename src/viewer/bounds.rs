use super::vertex::VertexPos;
use super::{pipeline::create_render_pipeline, shaders};
use crate::common::bounds::Bounds3;
use itertools::Itertools;

pub struct BoundsHandle {
    pub vertex_buffer: wgpu::Buffer,
    pub num_elements: usize,
}

impl BoundsHandle {
    pub fn from_bounds(device: &wgpu::Device, bounds: &Bounds3) -> Self {
        let min_xyz = bounds.p_min;
        let max_xyz = bounds.p_max;

        let min_xy_max_z = na::Point3::new(min_xyz.x, min_xyz.y, max_xyz.z);
        let min_xz_max_y = na::Point3::new(min_xyz.x, max_xyz.y, min_xyz.z);
        let min_yz_max_x = na::Point3::new(max_xyz.x, min_xyz.y, min_xyz.z);

        let min_z_max_xy = na::Point3::new(max_xyz.x, max_xyz.y, min_xyz.z);
        let min_y_max_xz = na::Point3::new(max_xyz.x, min_xyz.y, max_xyz.z);
        let min_x_max_yz = na::Point3::new(min_xyz.x, max_xyz.y, max_xyz.z);

        let line_list = [
            VertexPos::from(&min_xyz),
            VertexPos::from(&min_yz_max_x),
            VertexPos::from(&min_yz_max_x),
            VertexPos::from(&min_z_max_xy),
            VertexPos::from(&min_z_max_xy),
            VertexPos::from(&min_xz_max_y),
            VertexPos::from(&min_xz_max_y),
            VertexPos::from(&min_xyz),
            VertexPos::from(&min_xy_max_z),
            VertexPos::from(&min_y_max_xz),
            VertexPos::from(&min_y_max_xz),
            VertexPos::from(&max_xyz),
            VertexPos::from(&max_xyz),
            VertexPos::from(&min_x_max_yz),
            VertexPos::from(&min_x_max_yz),
            VertexPos::from(&min_xy_max_z),
            VertexPos::from(&min_xyz),
            VertexPos::from(&min_xy_max_z),
            VertexPos::from(&min_yz_max_x),
            VertexPos::from(&min_y_max_xz),
            VertexPos::from(&min_z_max_xy),
            VertexPos::from(&max_xyz),
            VertexPos::from(&min_xz_max_y),
            VertexPos::from(&min_x_max_yz),
        ];

        let vertex_buffer = device
            .create_buffer_with_data(bytemuck::cast_slice(&line_list), wgpu::BufferUsage::VERTEX);

        BoundsHandle {
            vertex_buffer,
            num_elements: line_list.len(),
        }
    }
}

pub struct BoundsRenderPass {
    render_pipeline: wgpu::RenderPipeline,
    bounds_handles: Vec<BoundsHandle>,
}

impl BoundsRenderPass {
    pub fn from_bounds(
        device: &wgpu::Device,
        mut compiler: &mut shaderc::Compiler,
        uniform_bind_group_layout: &wgpu::BindGroupLayout,
        bounds: &Vec<Bounds3>,
    ) -> Self {
        let (vs_module, fs_module) = shaders::flat::compile_shaders(&mut compiler, &device);

        let bounds_handles = bounds
            .iter()
            .map(|bounds| BoundsHandle::from_bounds(&device, bounds))
            .collect_vec();

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                bind_group_layouts: &[&uniform_bind_group_layout],
            });

        let render_pipeline = create_render_pipeline::<VertexPos>(
            &device,
            render_pipeline_layout,
            &vs_module,
            &fs_module,
            wgpu::PrimitiveTopology::LineList,
            true,
        );

        BoundsRenderPass {
            bounds_handles,
            render_pipeline,
        }
    }
}

pub trait DrawBounds<'a, 'b>
where
    'b: 'a,
{
    fn draw_bounds(&mut self, bounds: &'b BoundsHandle);
    fn draw_all_bounds(&mut self, bounds: &'b BoundsRenderPass);
}

impl<'a, 'b> DrawBounds<'a, 'b> for wgpu::RenderPass<'a>
where
    'b: 'a,
{
    fn draw_bounds(&mut self, bounds: &'b BoundsHandle) {
        self.set_vertex_buffer(0, &bounds.vertex_buffer, 0, 0);
        self.draw(0..bounds.num_elements as u32, 0..1);
    }

    fn draw_all_bounds(&mut self, bounds: &'b BoundsRenderPass) {
        self.set_pipeline(&bounds.render_pipeline);

        for bounds_handle in &bounds.bounds_handles {
            self.draw_bounds(&bounds_handle);
        }
    }
}
