mod texture;
mod camera;
mod shaders;

use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
    dpi::{LogicalPosition, PhysicalPosition},
};
use crate::common::{World, Camera, Mesh};
use camera::OrbitalCameraController;
use itertools::{interleave, zip_eq, Itertools};

lazy_static::lazy_static! {
    #[rustfmt::skip]
    static ref OPENGL_TO_WGPU_MATRIX: glm::Mat4 = glm::mat4(
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 0.5, 0.0,
        0.0, 0.0, 0.5, 1.0,
    );
}

pub trait Vertex {
    fn desc<'a>() -> wgpu::VertexBufferDescriptor<'a>;
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct DrawVertex {
    position: glm::Vec3,
    normal: glm::Vec3,
}

unsafe impl bytemuck::Zeroable for DrawVertex {}

unsafe impl bytemuck::Pod for DrawVertex {}

impl From<(&glm::Vec3, &glm::Vec3)> for DrawVertex {
    fn from(pair: (&glm::Vec3, &glm::Vec3)) -> Self {
        let (position, normal) = pair;
        DrawVertex { position: position.clone(), normal: normal.clone() }
    }
}

impl Vertex for DrawVertex {
    fn desc<'a>() -> wgpu::VertexBufferDescriptor<'a> {
        wgpu::VertexBufferDescriptor {
            stride: std::mem::size_of::<DrawVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::InputStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttributeDescriptor {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float3,
                },
                wgpu::VertexAttributeDescriptor {
                    offset: std::mem::size_of::<glm::Vec3>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float3,
                },
            ],
        }
    }
}

pub struct DrawMesh {
    pub index: usize,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub num_elements: usize,
}

impl DrawMesh {
    pub fn from_mesh(device: &wgpu::Device, mesh: &Mesh) -> Self {
        let vertices = zip_eq(&mesh.pos, &mesh.normal).map_into::<DrawVertex>().collect_vec();

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
    pub fn create_bind_group_layout_entry(device: &wgpu::Device) -> wgpu::BindGroupLayoutEntry {
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
    pub fn from_world(device: &wgpu::Device, instances_bind_group_layout: &wgpu::BindGroupLayout, world: &World, mesh: DrawMesh) -> Self {
        let instance_data = world.objects.iter().filter(|obj| obj.mesh.index == mesh.index).map(|obj| Instance { model: obj.obj_to_world.clone() }).collect_vec();

        let instance_buffer_size = instance_data.len() * std::mem::size_of::<glm::Mat4>();
        let instance_buffer = device.create_buffer_with_data(
            bytemuck::cast_slice(&instance_data[..]),
            wgpu::BufferUsage::STORAGE_READ,
        );

        let instances_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &instances_bind_group_layout,
            bindings: &[
                wgpu::Binding {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer {
                        buffer: &instance_buffer,
                        range: 0..instance_buffer_size as wgpu::BufferAddress,
                    },
                },
            ],
            label: Some("instances_bind_group"),
        });

        DrawMeshInstances {
            mesh,
            instance_buffer,
            instance_buffer_size,
            instances_bind_group,
            visible_instances: std::ops::Range { start: 0, end: instance_data.len() as u32 },
        }
    }
}

pub trait DrawModel<'a, 'b>
    where
        'b: 'a,
{
    fn draw_mesh_instances(
        &mut self,
        mesh: &'b DrawMeshInstances,
    );
}

impl<'a, 'b> DrawModel<'a, 'b> for wgpu::RenderPass<'a>
    where
        'b: 'a,
{
    fn draw_mesh_instances(
        &mut self,
        mesh_instances: &'b DrawMeshInstances,
    ) {
        self.set_bind_group(1, &mesh_instances.instances_bind_group, &[]);
        self.set_vertex_buffer(0, &mesh_instances.mesh.vertex_buffer, 0, 0);
        self.set_index_buffer(&mesh_instances.mesh.index_buffer, 0, 0);
        self.draw_indexed(0..mesh_instances.mesh.num_elements as u32, 0, mesh_instances.visible_instances.clone());
    }
}


#[repr(C)] // We need this for Rust to store our data correctly for the shaders
#[derive(Debug, Copy, Clone)] // This is so we can store this in a buffer
struct Uniforms {
    view_proj: glm::Mat4,
}

unsafe impl bytemuck::Zeroable for Uniforms {}

unsafe impl bytemuck::Pod for Uniforms {}

impl Uniforms {
    fn new() -> Self {
        Self {
            view_proj: glm::Mat4::identity(),
        }
    }

    fn update_view_proj(&mut self, camera: &Camera) {
        self.view_proj = *OPENGL_TO_WGPU_MATRIX * &camera.cam_to_screen * glm::inverse(&camera.cam_to_world);
    }
}

pub struct Viewer {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    sc_desc: wgpu::SwapChainDescriptor,
    swap_chain: wgpu::SwapChain,
    render_pipeline: wgpu::RenderPipeline,
    draw_mesh_instances: Vec<DrawMeshInstances>,
    uniforms: Uniforms,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    depth_texture: texture::Texture,
    size: winit::dpi::PhysicalSize<u32>,
    world: World,
    camera_controller: OrbitalCameraController,
    last_mouse_pos: PhysicalPosition<f64>,
    mouse_pressed: bool,
}

impl Viewer {
    pub async fn new(window: &Window, scene_path: &str) -> Self {
        let world = World::from_gltf(scene_path);

        let camera_controller = OrbitalCameraController::new(glm::vec3(0.0, 0.0, 0.0), 50.0, 0.01);

        let size = window.inner_size();

        let surface = wgpu::Surface::create(window);

        let adapter = wgpu::Adapter::request(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::Default,
                compatible_surface: Some(&surface),
            },
            wgpu::BackendBit::PRIMARY, // Vulkan + Metal + DX12 + Browser WebGPU
        ).await.unwrap();

        println!("{:?}", adapter.get_info());

        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor {
            extensions: wgpu::Extensions {
                anisotropic_filtering: false,
            },
            limits: Default::default(),
        }).await;

        let sc_desc = wgpu::SwapChainDescriptor {
            usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
            format: wgpu::TextureFormat::Bgra8UnormSrgb,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
        };
        let swap_chain = device.create_swap_chain(&surface, &sc_desc);

        let mut compiler = shaderc::Compiler::new().unwrap();

        let (vs_module, fs_module) = shaders::phong::compile_phong_shaders(&mut compiler, &device);

        let mut uniforms = Uniforms::new();
        uniforms.update_view_proj(&world.camera);

        let uniform_buffer = device.create_buffer_with_data(
            bytemuck::cast_slice(&[uniforms]),
            wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
        );

        let uniform_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            bindings: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStage::VERTEX,
                    ty: wgpu::BindingType::UniformBuffer {
                        dynamic: false,
                    },
                },
            ],
            label: Some("uniform_bind_group_layout"),
        });

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &uniform_bind_group_layout,
            bindings: &[
                wgpu::Binding {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer {
                        buffer: &uniform_buffer,
                        // FYI: you can share a single buffer between bindings.
                        range: 0..std::mem::size_of_val(&uniforms) as wgpu::BufferAddress,
                    },
                },
            ],
            label: Some("uniform_bind_group"),
        });

        let instances_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            bindings: &[
                Instance::create_bind_group_layout_entry(&device),
            ],
            label: Some("instances_bind_group_layout"),
        });

        let draw_mesh_instances = world.meshes.iter().map(|mesh| DrawMeshInstances::from_world(&device, &instances_bind_group_layout, &world, DrawMesh::from_mesh(&device, &mesh))).collect_vec();

        let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            bind_group_layouts: &[&uniform_bind_group_layout, &instances_bind_group_layout],
        });

        let depth_texture = texture::Texture::create_depth_texture(&device, &sc_desc, "depth_texture");

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            layout: &render_pipeline_layout,
            vertex_stage: wgpu::ProgrammableStageDescriptor {
                module: &vs_module,
                entry_point: "main", // 1.
            },
            fragment_stage: Some(wgpu::ProgrammableStageDescriptor { // 2.
                module: &fs_module,
                entry_point: "main",
            }),
            rasterization_state: Some(wgpu::RasterizationStateDescriptor {
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: wgpu::CullMode::Back,
                depth_bias: 0,
                depth_bias_slope_scale: 0.0,
                depth_bias_clamp: 0.0,
            }),
            color_states: &[
                wgpu::ColorStateDescriptor {
                    format: sc_desc.format,
                    color_blend: wgpu::BlendDescriptor::REPLACE,
                    alpha_blend: wgpu::BlendDescriptor::REPLACE,
                    write_mask: wgpu::ColorWrite::ALL,
                },
            ],
            primitive_topology: wgpu::PrimitiveTopology::TriangleList,
            depth_stencil_state: Some(wgpu::DepthStencilStateDescriptor {
                format: texture::Texture::DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less, // 1.
                stencil_front: wgpu::StencilStateFaceDescriptor::IGNORE,
                stencil_back: wgpu::StencilStateFaceDescriptor::IGNORE,
                stencil_read_mask: 0,
                stencil_write_mask: 0,
            }),
            vertex_state: wgpu::VertexStateDescriptor {
                index_format: wgpu::IndexFormat::Uint32,
                vertex_buffers: &[
                    DrawVertex::desc(),
                ],
            },
            sample_count: 1, // 5.
            sample_mask: !0, // 6.
            alpha_to_coverage_enabled: false, // 7.
        });

        Self {
            surface,
            device,
            queue,
            sc_desc,
            swap_chain,
            render_pipeline,
            draw_mesh_instances,
            uniforms,
            uniform_buffer,
            uniform_bind_group,
            depth_texture,
            size,
            world,
            camera_controller,
            last_mouse_pos: (0.0, 0.0).into(),
            mouse_pressed: false,
        }
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        self.size = new_size;
        self.sc_desc.width = new_size.width;
        self.sc_desc.height = new_size.height;
        self.depth_texture = texture::Texture::create_depth_texture(&self.device, &self.sc_desc, "depth_texture");
        self.swap_chain = self.device.create_swap_chain(&self.surface, &self.sc_desc);
    }

    // input() won't deal with GPU code, so it can be synchronous
    pub fn input(&mut self, event: &DeviceEvent) -> bool {
        match event {
            DeviceEvent::MouseWheel {
                delta,
                ..
            } => {
                self.camera_controller.process_scroll(delta);
                true
            }
            DeviceEvent::Button {
                button,
                state,
                ..
            } => {
                self.mouse_pressed = (*button == 1 && *state == ElementState::Pressed);
                true
            }
            DeviceEvent::MouseMotion {
                delta,
                ..
            } => {
                let (mouse_dx, mouse_dy) = delta;
                if self.mouse_pressed {
                    self.camera_controller.process_mouse(*mouse_dx, *mouse_dy);
                }
                true
            }
            _ => false,
        }
    }

    pub fn update(&mut self, dt: std::time::Duration) {
        self.camera_controller.update_camera(&mut self.world.camera, dt);
        self.uniforms.update_view_proj(&self.world.camera);

        // Copy operation's are performed on the gpu, so we'll need
        // a CommandEncoder for that
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("update encoder"),
        });

        let staging_buffer = self.device.create_buffer_with_data(
            bytemuck::cast_slice(&[self.uniforms]),
            wgpu::BufferUsage::COPY_SRC,
        );

        encoder.copy_buffer_to_buffer(&staging_buffer, 0, &self.uniform_buffer, 0, std::mem::size_of::<Uniforms>() as wgpu::BufferAddress);

        // We need to remember to submit our CommandEncoder's output
        // otherwise we won't see any change.
        self.queue.submit(&[encoder.finish()]);
    }

    pub fn render(&mut self) {
        let frame = self.swap_chain.get_next_texture()
            .expect("Timeout getting texture");
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[
                    wgpu::RenderPassColorAttachmentDescriptor {
                        attachment: &frame.view,
                        resolve_target: None,
                        load_op: wgpu::LoadOp::Clear,
                        store_op: wgpu::StoreOp::Store,
                        clear_color: wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        },
                    }
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachmentDescriptor {
                    attachment: &self.depth_texture.view,
                    depth_load_op: wgpu::LoadOp::Clear,
                    depth_store_op: wgpu::StoreOp::Store,
                    clear_depth: 1.0,
                    stencil_load_op: wgpu::LoadOp::Clear,
                    stencil_store_op: wgpu::StoreOp::Store,
                    clear_stencil: 0,
                }),
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);

            for mesh_instance in &self.draw_mesh_instances {
                render_pass.draw_mesh_instances(&mesh_instance);
            }
        }

        self.queue.submit(&[
            encoder.finish()
        ]);
    }
}
