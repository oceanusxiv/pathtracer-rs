use super::{pipeline::create_render_pipeline, shaders, texture, vertex::VertexPosTex};
use wgpu::util::DeviceExt;

const DEPTH_VERTICES: &[VertexPosTex] = &[
    VertexPosTex {
        position: [-1.0, -1.0, 0.0],
        tex_coords: [0.0, 1.0],
    },
    VertexPosTex {
        position: [1.0, -1.0, 0.0],
        tex_coords: [1.0, 1.0],
    },
    VertexPosTex {
        position: [1.0, 1.0, 0.0],
        tex_coords: [1.0, 0.0],
    },
    VertexPosTex {
        position: [-1.0, 1.0, 0.0],
        tex_coords: [0.0, 0.0],
    },
];

const DEPTH_INDICES: &[u32] = &[0, 1, 2, 0, 2, 3];

pub struct QuadHandle {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub texture: texture::Texture,
    pub texture_bind_group: wgpu::BindGroup,
    pub num_elements: usize,
}

impl QuadHandle {
    pub fn from_texture(
        device: &wgpu::Device,
        texture_bind_group_layout: &wgpu::BindGroupLayout,
        texture: texture::Texture,
    ) -> Self {
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(DEPTH_VERTICES),
            usage: wgpu::BufferUsage::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(DEPTH_INDICES),
            usage: wgpu::BufferUsage::INDEX,
        });

        let texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&texture.sampler),
                },
            ],
            label: Some("depth_pass.bind_group"),
        });

        QuadHandle {
            vertex_buffer,
            index_buffer,
            texture,
            texture_bind_group,
            num_elements: 6,
        }
    }
}

pub struct QuadRenderPass {
    render_pipeline: wgpu::RenderPipeline,
    pub quad: QuadHandle,
}

impl QuadRenderPass {
    pub fn from_texture(
        device: &wgpu::Device,
        compiler: &mut shaderc::Compiler,
        texture: texture::Texture,
    ) -> Self {
        let (vs_module, fs_module) = shaders::quad::compile_shaders(compiler, &device);

        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStage::FRAGMENT,
                        ty: wgpu::BindingType::SampledTexture {
                            multisampled: false,
                            dimension: wgpu::TextureViewDimension::D2,
                            component_type: wgpu::TextureComponentType::Uint,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStage::FRAGMENT,
                        ty: wgpu::BindingType::Sampler { comparison: false },
                        count: None,
                    },
                ],
                label: Some("texture_bind_group_layout"),
            });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&texture_bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline = create_render_pipeline::<VertexPosTex>(
            &device,
            render_pipeline_layout,
            &vs_module,
            &fs_module,
            wgpu::PrimitiveTopology::TriangleList,
            false,
        );

        QuadRenderPass {
            render_pipeline,
            quad: QuadHandle::from_texture(&device, &texture_bind_group_layout, texture),
        }
    }
}

pub trait DrawQuad<'a, 'b>
where
    'b: 'a,
{
    fn draw_quad(&mut self, quad: &'b QuadRenderPass);
}

impl<'a, 'b> DrawQuad<'a, 'b> for wgpu::RenderPass<'a>
where
    'b: 'a,
{
    fn draw_quad(&mut self, quad: &'b QuadRenderPass) {
        self.set_pipeline(&quad.render_pipeline);
        self.set_bind_group(0, &quad.quad.texture_bind_group, &[]);
        self.set_vertex_buffer(0, quad.quad.vertex_buffer.slice(..));
        self.set_index_buffer(quad.quad.index_buffer.slice(..));
        self.draw_indexed(0..quad.quad.num_elements as u32, 0, 0..1);
    }
}
