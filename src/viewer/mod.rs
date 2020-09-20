mod bounds;
pub mod camera;
pub mod importer;
mod mesh;
mod pipeline;
mod quad;
mod shaders;
mod texture;
mod vertex;
mod wireframe;

use crate::common::{bounds::Bounds3, Camera};
use bounds::{BoundsRenderPass, DrawBounds};
use camera::{CameraController, CameraControllerInterface};
use mesh::{DrawMesh, MeshRenderPass};
use quad::{DrawQuad, QuadRenderPass};
use wgpu::util::DeviceExt;
use winit::{event::*, window::Window};
use wireframe::{DrawWireFrame, WireFrameRenderPass};

lazy_static::lazy_static! {
    #[rustfmt::skip]
    static ref OPENGL_TO_WGPU_MATRIX: glm::Mat4 = glm::mat4(
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 0.5, 0.5,
        0.0, 0.0, 0.0, 1.0,
    );
}

pub struct Mesh {
    pub id: usize,
    pub indices: Vec<u32>,
    pub pos: Vec<na::Point3<f32>>,
    pub normal: Vec<na::Vector3<f32>>,
    pub s: Vec<na::Vector3<f32>>,
    pub uv: Vec<na::Point2<f32>>,
    pub colors: Vec<na::Vector3<f32>>,

    pub instances: Vec<na::Projective3<f32>>,
}
pub struct ViewerScene {
    meshes: Vec<Mesh>,
}

#[repr(C)] // We need this for Rust to store our data correctly for the shaders
#[derive(Debug, Copy, Clone)] // This is so we can store this in a buffer
struct Uniforms {
    view_proj: glm::Mat4,
}

unsafe impl bytemuck::Zeroable for Uniforms {}

unsafe impl bytemuck::Pod for Uniforms {}

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
                dynamic: false,
                readonly: true,
                min_binding_size: None,
            },
            count: None,
        }
    }
}

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
            ty: wgpu::BindingType::UniformBuffer {
                dynamic: false,
                min_binding_size: None,
            },
            count: None,
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
    wireframe_render_pass: WireFrameRenderPass,
    uniforms: Uniforms,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    depth_texture: texture::Texture,
    size: winit::dpi::PhysicalSize<u32>,
    camera_controller: CameraController,
    mouse_pressed: bool,
    pub state: ViewerState,
    pub draw_wireframe: bool,
    pub draw_mesh: bool,
    pub draw_bounds: bool,
    pub bounds_loaded: bool,
}

impl Viewer {
    pub async fn new(
        log: &slog::Logger,
        window: &Window,
        scene: &ViewerScene,
        camera: &Camera,
        camera_controller: CameraController,
    ) -> Self {
        let log = log.new(o!("module" => "viewer"));

        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::BackendBit::PRIMARY);
        let surface = unsafe { instance.create_surface(window) };
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::Default,
                compatible_surface: Some(&surface),
            })
            .await
            .unwrap();

        debug!(log, "{:?}", adapter.get_info());

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::default(),
                    shader_validation: true,
                },
                None,
            )
            .await
            .unwrap();

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

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
        });

        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[Uniforms::create_bind_group_layout_entry()],
                label: Some("uniform_bind_group_layout"),
            });

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(uniform_buffer.slice(..)),
            }],
            label: Some("uniform_bind_group"),
        });

        let mesh_render_pass =
            MeshRenderPass::from_scene(&device, &mut compiler, &uniform_bind_group_layout, &scene);

        let bounds_render_pass = BoundsRenderPass::from_bounds(
            &device,
            &mut compiler,
            &uniform_bind_group_layout,
            &vec![],
        );

        let wireframe_render_pass = WireFrameRenderPass::from_scene(
            &device,
            &mut compiler,
            &uniform_bind_group_layout,
            &scene,
        );

        let depth_texture =
            texture::Texture::create_depth_texture(&device, &sc_desc, "depth_texture");

        let rendered_texture = texture::Texture::from_image(
            &device,
            &queue,
            &camera.film.copy_image(),
            Some("rendered_texture"),
        )
        .unwrap();

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
            wireframe_render_pass,
            uniforms,
            uniform_buffer,
            uniform_bind_group,
            depth_texture,
            size,
            camera_controller,
            mouse_pressed: false,
            state: ViewerState::RenderScene,
            draw_wireframe: false,
            draw_mesh: true,
            draw_bounds: false,
            bounds_loaded: false,
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

    pub fn window_input(&mut self, event: &WindowEvent) -> bool {
        match self.state {
            ViewerState::RenderScene => match event {
                WindowEvent::KeyboardInput { input, .. } => match input {
                    KeyboardInput {
                        state: ElementState::Pressed,
                        virtual_keycode,
                        ..
                    } => {
                        if let Some(keycode) = virtual_keycode {
                            self.camera_controller.process_key(keycode)
                        } else {
                            false
                        }
                    }
                    _ => false,
                },
                _ => false,
            },
            _ => false,
        }
    }

    // input() won't deal with GPU code, so it can be synchronous
    pub fn device_input(&mut self, event: &DeviceEvent) -> bool {
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
                    if (self.camera_controller.require_mouse_press() && self.mouse_pressed)
                        || !self.camera_controller.require_mouse_press()
                    {
                        self.camera_controller.process_mouse(
                            mouse_dx / self.size.width as f64,
                            mouse_dy / self.size.height as f64,
                        );
                    }
                    true
                }
                _ => false,
            },
            _ => false,
        }
    }

    pub fn update_rendered_texture(&mut self, camera: &Camera) {
        let img = camera.film.copy_image();
        let dimensions = img.dimensions();

        let size = wgpu::Extent3d {
            width: dimensions.0,
            height: dimensions.1,
            depth: 1,
        };

        self.queue.write_texture(
            wgpu::TextureCopyView {
                texture: &self.quad_render_pass.quad.texture.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            &img,
            wgpu::TextureDataLayout {
                offset: 0,
                bytes_per_row: 4 * dimensions.0,
                rows_per_image: dimensions.1,
            },
            size,
        );
    }

    pub fn update_camera(&mut self, camera: &mut Camera, dt: std::time::Duration) {
        self.camera_controller.update_camera(camera, dt);
        self.uniforms.update_view_proj(camera);

        self.queue.write_buffer(
            &self.uniform_buffer,
            0,
            &bytemuck::cast_slice(&[self.uniforms]),
        );
    }

    pub fn update_bounds(&mut self, bounds: &Vec<Bounds3>) {
        if !self.bounds_loaded {
            self.bounds_render_pass.update_bounds(&self.device, &bounds);
            self.bounds_loaded = true;
        }
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
            .get_current_frame()
            .expect("Timeout getting texture")
            .output;

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
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: true,
                    },
                }],
                depth_stencil_attachment: None,
            });
            render_pass.draw_quad(&self.quad_render_pass);
        }

        self.queue.submit(Some(encoder.finish()));
    }

    pub fn render_scene(&mut self) {
        let frame = self
            .swap_chain
            .get_current_frame()
            .expect("Timeout getting texture")
            .output;

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
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.5,
                            g: 0.5,
                            b: 0.5,
                            a: 1.0,
                        }),
                        store: true,
                    },
                }],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachmentDescriptor {
                    attachment: &self.depth_texture.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: true,
                    }),
                    stencil_ops: None,
                }),
            });

            render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            if self.draw_mesh {
                render_pass.draw_all_mesh(&self.mesh_render_pass);
            }
            if self.draw_bounds {
                render_pass.draw_all_bounds(&self.bounds_render_pass);
            }
            if self.draw_wireframe {
                render_pass.draw_all_wire_frame(&self.wireframe_render_pass);
            }
        }

        self.queue.submit(Some(encoder.finish()));
    }
}
