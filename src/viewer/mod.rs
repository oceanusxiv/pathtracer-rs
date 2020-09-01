mod bounds;
mod camera;
mod mesh;
mod pipeline;
mod quad;
mod shaders;
mod texture;
mod vertex;

use crate::common::bounds::Bounds3;
use crate::common::{Camera, World};
use bounds::{BoundsRenderPass, DrawBounds};
use camera::OrbitalCameraController;
use mesh::{DrawMesh, MeshRenderPass};
use quad::{DrawQuad, QuadRenderPass};
use winit::{event::*, window::Window};

lazy_static::lazy_static! {
    #[rustfmt::skip]
    static ref OPENGL_TO_WGPU_MATRIX: glm::Mat4 = glm::mat4(
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 0.5, 0.0,
        0.0, 0.0, 0.5, 1.0,
    );
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
        self.view_proj = *OPENGL_TO_WGPU_MATRIX
            * (camera.cam_to_screen.to_projective() * camera.cam_to_world.inverse())
                .to_homogeneous();
    }

    pub fn create_bind_group_layout_entry() -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStage::VERTEX,
            ty: wgpu::BindingType::UniformBuffer { dynamic: false },
        }
    }
}

pub enum ViewerState {
    RenderScene,
    RenderImage,
}

pub struct Viewer {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    sc_desc: wgpu::SwapChainDescriptor,
    swap_chain: wgpu::SwapChain,
    mesh_render_pass: MeshRenderPass,
    bounds_render_pass: BoundsRenderPass,
    quad_render_pass: QuadRenderPass,
    uniforms: Uniforms,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    depth_texture: texture::Texture,
    size: winit::dpi::PhysicalSize<u32>,
    camera_controller: OrbitalCameraController,
    mouse_pressed: bool,
    pub state: ViewerState,
}

impl Viewer {
    pub async fn new(window: &Window, world: &World, camera: &Camera) -> Self {
        let camera_controller = OrbitalCameraController::new(glm::vec3(0.0, 0.0, 0.0), 50.0, 0.01);

        let size = window.inner_size();

        let surface = wgpu::Surface::create(window);

        let adapter = wgpu::Adapter::request(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::Default,
                compatible_surface: Some(&surface),
            },
            wgpu::BackendBit::PRIMARY, // Vulkan + Metal + DX12 + Browser WebGPU
        )
        .await
        .unwrap();

        debug!("{:?}", adapter.get_info());

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                extensions: wgpu::Extensions {
                    anisotropic_filtering: false,
                },
                limits: Default::default(),
            })
            .await;

        let sc_desc = wgpu::SwapChainDescriptor {
            usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
            format: wgpu::TextureFormat::Bgra8UnormSrgb,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
        };
        let swap_chain = device.create_swap_chain(&surface, &sc_desc);

        let mut compiler = shaderc::Compiler::new().unwrap();

        let mut uniforms = Uniforms::new();
        uniforms.update_view_proj(&camera);

        let uniform_buffer = device.create_buffer_with_data(
            bytemuck::cast_slice(&[uniforms]),
            wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
        );

        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                bindings: &[Uniforms::create_bind_group_layout_entry()],
                label: Some("uniform_bind_group_layout"),
            });

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &uniform_bind_group_layout,
            bindings: &[wgpu::Binding {
                binding: 0,
                resource: wgpu::BindingResource::Buffer {
                    buffer: &uniform_buffer,
                    // FYI: you can share a single buffer between bindings.
                    range: 0..std::mem::size_of_val(&uniforms) as wgpu::BufferAddress,
                },
            }],
            label: Some("uniform_bind_group"),
        });

        let mesh_render_pass =
            MeshRenderPass::from_world(&device, &mut compiler, &uniform_bind_group_layout, &world);

        let bounds_render_pass = BoundsRenderPass::from_bounds(
            &device,
            &mut compiler,
            &uniform_bind_group_layout,
            &vec![],
        );

        let depth_texture =
            texture::Texture::create_depth_texture(&device, &sc_desc, "depth_texture");

        let (rendered_texture, rendered_command_buffer) = texture::Texture::from_image(
            &device,
            &camera.film.copy_image(),
            Some("rendered_texture"),
        )
        .unwrap();
        queue.submit(&[rendered_command_buffer]);

        let quad_render_pass =
            QuadRenderPass::from_texture(&device, &mut compiler, rendered_texture);

        Self {
            surface,
            device,
            queue,
            sc_desc,
            swap_chain,
            mesh_render_pass,
            bounds_render_pass,
            quad_render_pass,
            uniforms,
            uniform_buffer,
            uniform_bind_group,
            depth_texture,
            size,
            camera_controller,
            mouse_pressed: false,
            state: ViewerState::RenderScene,
        }
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        self.size = new_size;
        self.sc_desc.width = new_size.width;
        self.sc_desc.height = new_size.height;
        self.depth_texture =
            texture::Texture::create_depth_texture(&self.device, &self.sc_desc, "depth_texture");
        self.swap_chain = self.device.create_swap_chain(&self.surface, &self.sc_desc);
    }

    // input() won't deal with GPU code, so it can be synchronous
    pub fn input(&mut self, event: &DeviceEvent) -> bool {
        match self.state {
            ViewerState::RenderScene => match event {
                DeviceEvent::MouseWheel { delta, .. } => {
                    self.camera_controller.process_scroll(delta);
                    true
                }
                DeviceEvent::Button { button, state, .. } => {
                    self.mouse_pressed =
                        (*button == 0 || *button == 1) && *state == ElementState::Pressed;
                    true
                }
                DeviceEvent::MouseMotion { delta, .. } => {
                    let (mouse_dx, mouse_dy) = delta;
                    if self.mouse_pressed {
                        self.camera_controller.process_mouse(*mouse_dx, *mouse_dy);
                    }
                    true
                }
                _ => false,
            },
            _ => false,
        }
    }

    pub fn update_rendered_texture(&mut self, camera: &Camera) {
        let cmd = texture::Texture::get_image_to_texture_cmd(
            &self.device,
            &camera.film.copy_image(),
            &self.quad_render_pass.quad.texture.texture,
        );

        self.queue.submit(&[cmd]);
    }

    pub fn update_camera(&mut self, camera: &mut Camera, dt: std::time::Duration) {
        self.camera_controller.update_camera(camera, dt);
        self.uniforms.update_view_proj(camera);

        // Copy operation's are performed on the gpu, so we'll need
        // a CommandEncoder for that
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("update encoder"),
            });

        let staging_buffer = self.device.create_buffer_with_data(
            bytemuck::cast_slice(&[self.uniforms]),
            wgpu::BufferUsage::COPY_SRC,
        );

        encoder.copy_buffer_to_buffer(
            &staging_buffer,
            0,
            &self.uniform_buffer,
            0,
            std::mem::size_of::<Uniforms>() as wgpu::BufferAddress,
        );

        // We need to remember to submit our CommandEncoder's output
        // otherwise we won't see any change.
        self.queue.submit(&[encoder.finish()]);
    }

    pub fn render(&mut self) {
        match self.state {
            ViewerState::RenderScene => {
                self.render_scene();
            }
            ViewerState::RenderImage => {
                self.render_image();
            }
        }
    }

    pub fn render_image(&mut self) {
        let frame = self
            .swap_chain
            .get_next_texture()
            .expect("Timeout getting texture");
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                    attachment: &frame.view,
                    resolve_target: None,
                    load_op: wgpu::LoadOp::Clear,
                    store_op: wgpu::StoreOp::Store,
                    clear_color: wgpu::Color::BLACK,
                }],
                depth_stencil_attachment: None,
            });
            render_pass.draw_quad(&self.quad_render_pass);
        }

        self.queue.submit(&[encoder.finish()]);
    }

    pub fn render_scene(&mut self) {
        let frame = self
            .swap_chain
            .get_next_texture()
            .expect("Timeout getting texture");
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                    attachment: &frame.view,
                    resolve_target: None,
                    load_op: wgpu::LoadOp::Clear,
                    store_op: wgpu::StoreOp::Store,
                    clear_color: wgpu::Color {
                        r: 0.5,
                        g: 0.5,
                        b: 0.5,
                        a: 1.0,
                    },
                }],
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

            render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            render_pass.draw_all_mesh(&self.mesh_render_pass);
            render_pass.draw_all_bounds(&self.bounds_render_pass);
        }

        self.queue.submit(&[encoder.finish()]);
    }
}
