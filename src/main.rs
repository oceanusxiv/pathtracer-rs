#[macro_use]
extern crate slog;

extern crate nalgebra as na;

use clap::clap_app;
use pathtracer_rs::*;
use slog::Drain;
use std::path::Path;
use winit::{
    dpi::{LogicalSize, Size},
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

fn sample_arg_legal(val: String) -> Result<(), String> {
    if let Ok(val) = val.parse::<f64>() {
        if val.sqrt() % 1.0 == 0.0 {
            Ok(())
        } else {
            Err(String::from("arg is not perfect square"))
        }
    } else {
        Err(String::from("could not parse arg samples"))
    }
}

fn new_drain(level: slog::Level) -> slog::Fuse<slog::LevelFilter<slog::Fuse<slog_async::Async>>> {
    let decorator = slog_term::TermDecorator::new().build();
    let drain = slog_term::FullFormat::new(decorator).build().fuse();
    let drain = slog_async::Async::new(drain).build().fuse();
    drain.filter_level(level).fuse()
}

fn main() {
    let info_drain = new_drain(slog::Level::Info);
    let drain = slog_atomic::AtomicSwitch::new(info_drain);
    let ctrl = drain.ctrl();
    let log = slog::Logger::root(drain.fuse(), o!());
    let mut trace_mode = false;

    let matches = clap_app!(pathtracer_rs =>
        (version: "1.0")
        (author: "Eric F. <eric1221bday@gmail.com>")
        (about: "Rust path tracer")
        (@arg SCENE: +required "Sets the input scene to use")
        (@arg output: -o --output +takes_value +required "Sets the output directory to save renders at")
        (@arg samples: -s --samples default_value("1") validator(sample_arg_legal) "Number of samples path tracer to take per pixel (must be perfect square)")
        (@arg verbose: -v --verbose "Print test information verbosely")
    )
    .get_matches();

    let scene_path = matches.value_of("SCENE").unwrap();
    let output_path = Path::new(matches.value_of("output").unwrap()).join("render.png");
    let pixel_samples_sqrt = matches
        .value_of("samples")
        .unwrap()
        .parse::<usize>()
        .unwrap();
    let (world, mut camera) = common::World::from_gltf(scene_path);
    let render_scene = pathtracer::RenderScene::from_world(&log, &world);
    let sampler =
        pathtracer::sampling::Sampler::new(pixel_samples_sqrt, pixel_samples_sqrt, true, 8);
    let integrator = pathtracer::DirectLightingIntegrator::new(&log, sampler);

    debug!(log, "camera starting at: {:?}", camera.cam_to_world);

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_inner_size(Size::Logical(LogicalSize::new(
            common::DEFAULT_RESOLUTION.x as f64,
            common::DEFAULT_RESOLUTION.y as f64,
        )))
        .build(&event_loop)
        .unwrap();
    let mut viewer =
        futures::executor::block_on(viewer::Viewer::new(&log, &window, &world, &camera));

    let mut last_render_time = std::time::Instant::now();
    let mut cursor_in_window = true;
    let mut crtl_clicked = false;
    let mut cursor_position: winit::dpi::PhysicalPosition<f64> =
        winit::dpi::PhysicalPosition::new(0.0, 0.0);
    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        match event {
            Event::DeviceEvent {
                ref event,
                device_id: _,
            } => {
                if cursor_in_window {
                    viewer.input(event);
                }
            }
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == window.id() => {
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
                            integrator.render(&mut camera, &render_scene);
                            viewer.update_rendered_texture(&camera);
                            viewer.state = viewer::ViewerState::RenderImage
                        }
                        KeyboardInput {
                            state: ElementState::Pressed,
                            virtual_keycode: Some(VirtualKeyCode::C),
                            ..
                        } => viewer.state = viewer::ViewerState::RenderScene,
                        KeyboardInput {
                            state: ElementState::Pressed,
                            virtual_keycode: Some(VirtualKeyCode::W),
                            ..
                        } => viewer.draw_wireframe = !viewer.draw_wireframe,
                        KeyboardInput {
                            state: ElementState::Pressed,
                            virtual_keycode: Some(VirtualKeyCode::S),
                            ..
                        } => {
                            info!(log, "saving image to {:?}", &output_path);
                            camera.film.save(&output_path);
                        }
                        KeyboardInput {
                            state: ElementState::Pressed,
                            virtual_keycode: Some(VirtualKeyCode::T),
                            ..
                        } => {
                            if trace_mode {
                                info!(log, "setting log level to info");
                                ctrl.set(new_drain(slog::Level::Info));
                            } else {
                                info!(log, "setting log level to trace");
                                ctrl.set(new_drain(slog::Level::Trace));
                            }
                            trace_mode = !trace_mode;
                        }
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
                            integrator.render_single_pixel(&mut camera, pixel, &render_scene);
                        }
                    }
                    WindowEvent::CursorMoved { position, .. } => {
                        cursor_position = *position;
                    }
                    _ => {}
                }
            }
            Event::RedrawRequested(_) => {
                let now = std::time::Instant::now();
                let dt = now - last_render_time;
                last_render_time = now;
                viewer.update_camera(&mut camera, dt);
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
}
