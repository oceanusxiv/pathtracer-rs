mod bounds;
pub mod camera;
pub mod importer;
mod mesh;
mod pipeline;
mod quad;
pub mod renderer;
mod shaders;
mod texture;
mod vertex;
mod wireframe;

use crate::common::{new_drain, Camera};
use crate::pathtracer::{integrator::PathIntegrator, sampler::SamplerBuilder, RenderScene};
use crossbeam::scope;
use renderer::{Renderer, ViewerScene};
use std::sync::RwLock;
use std::{
    collections::{hash_map::RandomState, HashMap, HashSet},
    sync::atomic::AtomicBool,
    sync::atomic::Ordering,
};
use std::{path::PathBuf, time::Instant};
use winit::{
    dpi::{LogicalSize, Size},
    event::*,
    event_loop::{ControlFlow, EventLoop},
    platform::run_return::EventLoopExtRunReturn,
    window::WindowBuilder,
};

pub fn run(
    log: slog::Logger,
    resolution: &na::Vector2<f32>,
    viewer_scene: &ViewerScene,
    render_scene: RenderScene,
    camera: Camera,
    camera_controller_type: &str,
    integrator: PathIntegrator,
    output_path: PathBuf,
    ctrl: slog_atomic::AtomicSwitchCtrl,
    mut pixel_samples: usize,
    max_depth: i32,
    init_log_level: slog::Level,
    allowed_modules: Option<HashMap<String, HashSet<String, RandomState>, RandomState>>,
) {
    let camera = RwLock::new(camera);
    let integrator = RwLock::new(integrator);
    let camera_controller;
    if camera_controller_type == "orbit" {
        camera_controller = camera::CameraController::Orbit(camera::OrbitalCameraController::new(
            &log,
            na::Vector3::new(0.0, 0.0, 0.0),
            5000.0,
            0.01,
        ));
    } else if camera_controller_type == "fp" {
        camera_controller = camera::CameraController::FirstPerson(
            camera::FirstPersonCameraController::new(&log, 6000.0, 2.5),
        );
    } else {
        panic!(
            "invalid camera controller type: {:?}",
            camera_controller_type
        )
    }

    let mut event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("pathtracer-rs")
        .with_inner_size(Size::Logical(LogicalSize::new(
            resolution.x as f64,
            resolution.y as f64,
        )))
        .build(&event_loop)
        .unwrap();

    let mut viewer;
    {
        let camera = camera.read().unwrap();
        viewer = futures::executor::block_on(Renderer::new(
            &log,
            &window,
            &viewer_scene,
            &camera,
            camera_controller,
        ));
    }

    let mut last_render_time = Instant::now();
    let mut cursor_in_window = true;
    let mut crtl_clicked = false;
    let mut trace_mode = false;
    let mut cursor_position: winit::dpi::PhysicalPosition<f64> =
        winit::dpi::PhysicalPosition::new(0.0, 0.0);
    let (tx, rx) = crossbeam::channel::unbounded();

    scope(|s| {
        let render_closure = |_: &crossbeam::thread::Scope| {
            let rendering_done = AtomicBool::new(false);
            scope(|s| {
                s.spawn(|_| {
                    let camera = camera.read().unwrap();
                    while !rendering_done.load(Ordering::Relaxed) {
                        tx.send(camera.film.to_rgba_image()).unwrap();
                        std::thread::sleep(std::time::Duration::from_secs(2));
                    }

                    tx.send(camera.film.to_rgba_image()).unwrap();
                });

                let camera = camera.read().unwrap();
                let integrator = integrator.read().unwrap();

                integrator.render(&camera, &render_scene);
                rendering_done.store(true, Ordering::Relaxed);
            })
            .unwrap();
        };

        event_loop.run_return(|event, _, control_flow| {
            *control_flow = ControlFlow::Poll;
            match event {
                Event::DeviceEvent {
                    ref event,
                    device_id: _,
                } => {
                    if cursor_in_window {
                        viewer.device_input(event);
                    }
                }
                Event::WindowEvent {
                    ref event,
                    window_id,
                } if window_id == window.id() => {
                    if !viewer.window_input(event) {
                        match event {
                            WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                            WindowEvent::KeyboardInput { input, .. } => match input {
                                KeyboardInput {
                                    state: ElementState::Pressed,
                                    virtual_keycode: Some(VirtualKeyCode::Escape),
                                    ..
                                } => *control_flow = ControlFlow::Exit,
                                KeyboardInput {
                                    state: ElementState::Pressed,
                                    virtual_keycode: Some(VirtualKeyCode::R),
                                    ..
                                } => {
                                    let camera = camera.read().unwrap();
                                    camera.film.clear();
                                    viewer.state = renderer::ViewerState::RenderImage;
                                    s.spawn(render_closure);
                                }
                                KeyboardInput {
                                    state: ElementState::Pressed,
                                    virtual_keycode: Some(VirtualKeyCode::C),
                                    ..
                                } => viewer.state = renderer::ViewerState::RenderScene,
                                KeyboardInput {
                                    state: ElementState::Pressed,
                                    virtual_keycode: Some(VirtualKeyCode::G),
                                    ..
                                } => {
                                    if crtl_clicked {
                                        viewer.draw_wireframe = !viewer.draw_wireframe;
                                    }
                                }
                                KeyboardInput {
                                    state: ElementState::Pressed,
                                    virtual_keycode: Some(VirtualKeyCode::H),
                                    ..
                                } => {
                                    if crtl_clicked {
                                        viewer.draw_mesh = !viewer.draw_mesh;
                                    }
                                }
                                KeyboardInput {
                                    state: ElementState::Pressed,
                                    virtual_keycode: Some(VirtualKeyCode::B),
                                    ..
                                } => {
                                    if crtl_clicked {
                                        viewer.update_bounds(&render_scene.get_bounding_boxes());
                                        viewer.draw_bounds = !viewer.draw_bounds;
                                    }
                                }
                                KeyboardInput {
                                    state: ElementState::Pressed,
                                    virtual_keycode: Some(VirtualKeyCode::S),
                                    ..
                                } => {
                                    if crtl_clicked {
                                        info!(log, "saving image to {:?}", &output_path);
                                        let camera = camera.read().unwrap();
                                        camera.film.to_rgba_image().save(&output_path).unwrap();
                                    }
                                }
                                KeyboardInput {
                                    state: ElementState::Pressed,
                                    virtual_keycode: Some(VirtualKeyCode::T),
                                    ..
                                } => {
                                    if trace_mode {
                                        info!(log, "setting log level to {:?}", init_log_level);
                                        ctrl.set(new_drain(slog::Level::Info, &allowed_modules));
                                    } else {
                                        info!(log, "setting log level to {:?}", slog::Level::Trace);
                                        ctrl.set(new_drain(slog::Level::Trace, &allowed_modules));
                                    }
                                    trace_mode = !trace_mode;
                                }
                                KeyboardInput {
                                    state: ElementState::Pressed,
                                    virtual_keycode: Some(VirtualKeyCode::P),
                                    ..
                                } => match viewer.state {
                                    renderer::ViewerState::RenderScene => {
                                        let mut integrator = integrator.write().unwrap();
                                        integrator.toggle_progress_bar();
                                    }
                                    _ => {}
                                },
                                KeyboardInput {
                                    state: ElementState::Pressed,
                                    virtual_keycode: Some(VirtualKeyCode::Up),
                                    ..
                                } => match viewer.state {
                                    renderer::ViewerState::RenderScene => {
                                        pixel_samples *= 2;
                                        info!(
                                            log,
                                            "pixel sample increment now {:?}", pixel_samples
                                        );
                                        let camera = camera.read().unwrap();
                                        let mut integrator = integrator.write().unwrap();
                                        *integrator = PathIntegrator::new(
                                            &log,
                                            SamplerBuilder::new(
                                                &log,
                                                pixel_samples,
                                                &camera.film.get_sample_bounds(),
                                            ),
                                            max_depth as i32,
                                            true,
                                        );
                                        integrator.preprocess(&render_scene);
                                    }
                                    renderer::ViewerState::RenderImage => {}
                                },
                                KeyboardInput {
                                    state: ElementState::Pressed,
                                    virtual_keycode: Some(VirtualKeyCode::Down),
                                    ..
                                } => match viewer.state {
                                    renderer::ViewerState::RenderScene => {
                                        pixel_samples = 1.max(pixel_samples / 2);
                                        info!(
                                            log,
                                            "pixel sample increment now {:?}", pixel_samples
                                        );
                                        let camera = camera.read().unwrap();
                                        let mut integrator = integrator.write().unwrap();
                                        *integrator = PathIntegrator::new(
                                            &log,
                                            SamplerBuilder::new(
                                                &log,
                                                pixel_samples,
                                                &camera.film.get_sample_bounds(),
                                            ),
                                            max_depth as i32,
                                            true,
                                        );
                                        integrator.preprocess(&render_scene);
                                    }
                                    renderer::ViewerState::RenderImage => {}
                                },
                                _ => {}
                            },
                            WindowEvent::Resized(physical_size) => {
                                viewer.resize(*physical_size);
                            }
                            WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                                // new_inner_size is &mut so w have to dereference it twice
                                viewer.resize(**new_inner_size);
                            }
                            WindowEvent::CursorEntered { device_id: _ } => {
                                cursor_in_window = true;
                            }
                            WindowEvent::CursorLeft { device_id: _ } => {
                                cursor_in_window = false;
                            }
                            WindowEvent::ModifiersChanged(modifier) => match *modifier {
                                ModifiersState::CTRL => {
                                    crtl_clicked = true;
                                }
                                ModifiersState::LOGO => {
                                    crtl_clicked = true;
                                }
                                _ => {
                                    crtl_clicked = false;
                                }
                            },
                            WindowEvent::MouseInput {
                                state: ElementState::Released,
                                button: MouseButton::Left,
                                ..
                            } => {
                                if crtl_clicked {
                                    let pixel = na::Point2::new(
                                        (cursor_position.x / window.scale_factor()).floor() as i32,
                                        (cursor_position.y / window.scale_factor()).floor() as i32,
                                    );
                                    let camera = camera.read().unwrap();
                                    let integrator = integrator.read().unwrap();
                                    integrator.render_single_pixel(&camera, pixel, &render_scene);
                                }
                            }
                            WindowEvent::CursorMoved { position, .. } => {
                                cursor_position = *position;
                            }
                            _ => {}
                        }
                    }
                }
                Event::RedrawRequested(_) => {
                    let now = std::time::Instant::now();
                    let dt = now - last_render_time;
                    last_render_time = now;
                    viewer.update_camera(&camera, dt);

                    if let Ok(image) = rx.try_recv() {
                        viewer.update_rendered_texture(image);
                    }

                    viewer.render();
                }
                Event::MainEventsCleared => {
                    // RedrawRequested will only trigger once, unless we manually
                    // request it.
                    window.request_redraw();
                }
                _ => {}
            }
        });
    })
    .unwrap();
}
