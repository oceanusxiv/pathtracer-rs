use std::sync::RwLock;

use super::bounds::{BoundsRenderPass, DrawBounds};
use super::camera::{CameraController, CameraControllerInterface};
use super::mesh::{DrawMesh, MeshRenderPass};
use super::quad::{DrawQuad, QuadRenderPass};
use super::texture::Texture;
use super::wireframe::{DrawWireFrame, WireFrameRenderPass};
use crate::common::{bounds::Bounds3, Camera};
use wgpu::util::DeviceExt;
use winit::{event::*, window::Window};

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
    pub meshes: Vec<Mesh>,
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
    pub model: glm::Mat4,
}

unsafe impl bytemuck::Zeroable for Instance {}

unsafe impl bytemuck::Pod for Instance {}

impl Instance {
    pub fn create_bind_group_layout_entry() -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
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
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
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

pub struct Renderer {
    surface: wgpu::Surface,
    surface_config: wgpu::SurfaceConfiguration,
    device: wgpu::Device,
    queue: wgpu::Queue,
    mesh_render_pass: MeshRenderPass,
    bounds_render_pass: BoundsRenderPass,
    quad_render_pass: QuadRenderPass,
    wireframe_render_pass: WireFrameRenderPass,
    uniforms: Uniforms,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    depth_texture: Texture,
    size: winit::dpi::PhysicalSize<u32>,
    camera_controller: CameraController,
    mouse_pressed: bool,
    pub state: ViewerState,
    pub draw_wireframe: bool,
    pub draw_mesh: bool,
    pub draw_bounds: bool,
    pub bounds_loaded: bool,
}

impl Renderer {
    pub async fn new(
        log: &slog::Logger,
        window: &Window,
        scene: &ViewerScene,
        camera: &Camera,
        camera_controller: CameraController,
    ) -> Self {
        let log = log.new(o!("module" => "viewer"));

        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            dx12_shader_compiler: Default::default(),
        });
        let surface = unsafe { instance.create_surface(&window) }.unwrap();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        debug!(log, "{:?}", adapter.get_info());

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::default(),
                    label: None,
                },
                None,
            )
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .filter(|f| f.describe().srgb)
            .next()
            .unwrap_or(surface_caps.formats[0]);
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &surface_config);

        let mut compiler = shaderc::Compiler::new().unwrap();

        let mut uniforms = Uniforms::new();
        uniforms.update_view_proj(&camera);

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
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
                resource: uniform_buffer.as_entire_binding(),
            }],
            label: Some("uniform_bind_group"),
        });

        let mesh_render_pass = MeshRenderPass::from_scene(
            &device,
            &surface_config,
            &mut compiler,
            &uniform_bind_group_layout,
            &scene,
        );

        let bounds_render_pass = BoundsRenderPass::from_bounds(
            &device,
            &surface_config,
            &mut compiler,
            &uniform_bind_group_layout,
            &vec![],
        );

        let wireframe_render_pass = WireFrameRenderPass::from_scene(
            &device,
            &surface_config,
            &mut compiler,
            &uniform_bind_group_layout,
            &scene,
        );

        let depth_texture =
            Texture::create_depth_texture(&device, &surface_config, "depth_texture");

        let rendered_texture = Texture::from_image(
            &device,
            &queue,
            &image::RgbaImage::new(camera.film.resolution.x, camera.film.resolution.y),
            Some("rendered_texture"),
        )
        .unwrap();

        let quad_render_pass =
            QuadRenderPass::from_texture(&device, &surface_config, &mut compiler, rendered_texture);

        Self {
            surface,
            device,
            queue,
            surface_config,
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
        self.surface_config.width = new_size.width;
        self.surface_config.height = new_size.height;
        self.depth_texture =
            Texture::create_depth_texture(&self.device, &self.surface_config, "depth_texture");
        self.surface.configure(&self.device, &self.surface_config);
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

    pub fn update_rendered_texture(&mut self, img: image::RgbaImage) {
        let dimensions = img.dimensions();

        let size = wgpu::Extent3d {
            width: dimensions.0,
            height: dimensions.1,
            depth_or_array_layers: 1,
        };

        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.quad_render_pass.quad.texture.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &img,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: std::num::NonZeroU32::new(4 * dimensions.0),
                rows_per_image: std::num::NonZeroU32::new(dimensions.1),
            },
            size,
        );
    }

    pub fn update_camera(&mut self, camera: &RwLock<Camera>, dt: std::time::Duration) {
        match self.state {
            ViewerState::RenderScene => {
                let mut camera = camera.write().unwrap();
                self.camera_controller.update_camera(&mut camera, dt);
                self.uniforms.update_view_proj(&camera);

                self.queue.write_buffer(
                    &self.uniform_buffer,
                    0,
                    &bytemuck::cast_slice(&[self.uniforms]),
                );
            }
            _ => {}
        }
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

    pub fn render_image(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Image Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });
            render_pass.draw_quad(&self.quad_render_pass);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    pub fn render_scene(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Image Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
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
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture.view,
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

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}
