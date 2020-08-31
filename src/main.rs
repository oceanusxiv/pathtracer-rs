#![feature(new_uninit)]
#![feature(slice_partition_at_index)]
#![feature(iter_partition_in_place)]
#![feature(clamp)]

#[macro_use]
extern crate bitflags;

#[macro_use]
extern crate hexf;

extern crate nalgebra as na;
extern crate nalgebra_glm as glm;

mod common;
mod pathtracer;
mod viewer;

use clap::clap_app;
use std::path::Path;
use winit::{
    dpi::{LogicalSize, Size},
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

fn main() {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Warn)
        .filter_module("pathtracer_rs", log::LevelFilter::Info)
        .init();

    let matches = clap_app!(pathtracer_rs =>
        (version: "1.0")
        (author: "Eric F. <eric1221bday@gmail.com>")
        (about: "Rust path tracer")
        (@arg SCENE: +required "Sets the input scene to use")
        (@arg output: -o --output +takes_value +required "Sets the output directory to save renders at")
        (@arg verbose: -v --verbose "Print test information verbosely")
    )
    .get_matches();

    let scene_path = matches.value_of("SCENE").unwrap();
    let output_path = Path::new(matches.value_of("output").unwrap()).join("render.png");
    let (world, mut camera) = common::World::from_gltf(scene_path);
    let render_scene = pathtracer::RenderScene::from_world(&world);
    let sampler = pathtracer::sampling::Sampler::new(2, 2, true, 8);
    let integrator = pathtracer::DirectLightingIntegrator::new(sampler);

    print!("camera starting at: {:?}", camera.cam_to_world);

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_inner_size(Size::Logical(LogicalSize::new(
            common::DEFAULT_RESOLUTION.x as f64,
            common::DEFAULT_RESOLUTION.y as f64,
        )))
        .build(&event_loop)
        .unwrap();
    let mut viewer = futures::executor::block_on(viewer::Viewer::new(&window, &world, &camera));

    let mut last_render_time = std::time::Instant::now();
    let mut cursor_in_window = true;
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
                            virtual_keycode: Some(VirtualKeyCode::S),
                            ..
                        } => {
                            println!("saving image to {:?}", &output_path);
                            camera.film.save(&output_path);
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
                    WindowEvent::CursorEntered { device_id } => {
                        cursor_in_window = true;
                    }
                    WindowEvent::CursorLeft { device_id } => {
                        cursor_in_window = false;
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
