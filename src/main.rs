#[macro_use]
extern crate slog;

#[macro_use]
extern crate anyhow;

extern crate nalgebra as na;

use anyhow::Result;
use clap::clap_app;
use pathtracer_rs::*;
use slog::Drain;
use std::{time::Instant, path::Path};
use winit::{
    dpi::{LogicalSize, Size},
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

const MAX_DEPTH: u32 = 5;

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
    let drain = slog_async::Async::new(drain).chan_size(1000).build().fuse();
    drain.filter_level(level).fuse()
}

fn parse_resolution(res_str: &str) -> Result<na::Vector2<f32>> {
    let xy = res_str.split("x").collect::<Vec<_>>();
    if xy.len() != 2 {
        Err(anyhow!("invalid resolution string"))
    } else {
        let x = xy[0].parse::<f32>()?;
        let y = xy[1].parse::<f32>()?;

        Ok(na::Vector2::new(x, y))
    }
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
        (@arg resolution: -r --resolution +takes_value "Resolution of the window")
        (@arg max_depth: -d --max-depth +takes_value default_value("5") "Maximum ray tracing depth")
        (@arg verbose: -v --verbose "Print test information verbosely")
    )
    .get_matches();

    let scene_path = matches.value_of("SCENE").unwrap();
    let output_path = Path::new(matches.value_of("output").unwrap()).join("render.png");
    let mut pixel_samples_sqrt = matches
        .value_of("samples")
        .unwrap()
        .parse::<f64>()
        .unwrap()
        .sqrt() as usize;
    let resolution = if let Some(res_str) = matches.value_of("resolution") {
        parse_resolution(&res_str).unwrap_or_else(|_| {
            warn!(
                log,
                "failed parsing resolution string, falling back to default resolution"
            );
            *common::DEFAULT_RESOLUTION
        })
    } else {
        *common::DEFAULT_RESOLUTION
    };
    let max_depth = matches
        .value_of("max_depth")
        .unwrap()
        .parse::<u32>()
        .unwrap_or_else(|err| {
            warn!(
                log,
                "failed parsing max depth, falling back to default max depth"
            );
            MAX_DEPTH
        });

    let start = Instant::now();
    let (world, mut camera) = common::World::from_gltf(scene_path, &resolution);
    let render_scene = pathtracer::RenderScene::from_world(&log, &world);
    let sampler =
        pathtracer::sampling::Sampler::new(pixel_samples_sqrt, pixel_samples_sqrt, true, 8);
    let mut integrator =
        pathtracer::integrator::DirectLightingIntegrator::new(&log, sampler, max_depth);

    debug!(log, "camera starting at: {:?}", camera.cam_to_world);

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_inner_size(Size::Logical(LogicalSize::new(
            resolution.x as f64,
            resolution.y as f64,
        )))
        .build(&event_loop)
        .unwrap();
    let mut viewer =
        futures::executor::block_on(viewer::Viewer::new(&log, &window, &world, &camera));
    debug!(log, "initialization took: {:?}", start.elapsed());

    let mut last_render_time = Instant::now();
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
                        KeyboardInput {
                            state: ElementState::Pressed,
                            virtual_keycode: Some(VirtualKeyCode::Up),
                            ..
                        } => {
                            pixel_samples_sqrt += 1;
                            info!(
                                log,
                                "pixel sample count now {:?}",
                                pixel_samples_sqrt * pixel_samples_sqrt
                            );
                            integrator.set_sampler(pathtracer::sampling::Sampler::new(
                                pixel_samples_sqrt,
                                pixel_samples_sqrt,
                                true,
                                8,
                            ));
                        }
                        KeyboardInput {
                            state: ElementState::Pressed,
                            virtual_keycode: Some(VirtualKeyCode::Down),
                            ..
                        } => {
                            pixel_samples_sqrt = 1.max(pixel_samples_sqrt - 1);
                            info!(
                                log,
                                "pixel sample count now {:?}",
                                pixel_samples_sqrt * pixel_samples_sqrt
                            );
                            integrator.set_sampler(pathtracer::sampling::Sampler::new(
                                pixel_samples_sqrt,
                                pixel_samples_sqrt,
                                true,
                                8,
                            ));
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
