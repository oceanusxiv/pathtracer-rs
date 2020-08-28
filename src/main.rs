#![feature(new_uninit)]
#![feature(slice_partition_at_index)]
#![feature(iter_partition_in_place)]

extern crate nalgebra as na;
extern crate nalgebra_glm as glm;

mod common;
mod pathtracer;
mod viewer;

use clap::clap_app;
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
        (@arg verbose: -v --verbose "Print test information verbosely")
    )
    .get_matches();

    let scene_path = matches.value_of("SCENE").unwrap();
    let (world, mut camera) = common::World::from_gltf(scene_path);
    let render_scene = pathtracer::RenderScene::from_world(&world);
    let integrator = pathtracer::DirectLightingIntegrator::new();

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
    let mut window_focused = true;

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        match event {
            Event::DeviceEvent {
                ref event,
                device_id: _,
            } => {
                if window_focused {
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
                        } => integrator.render(
                            &mut camera,
                            &render_scene,
                            "/Users/eric/Downloads/duck.png",
                        ),
                        _ => {}
                    },
                    WindowEvent::Resized(physical_size) => {
                        viewer.resize(*physical_size);
                    }
                    WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                        // new_inner_size is &mut so w have to dereference it twice
                        viewer.resize(**new_inner_size);
                    }
                    WindowEvent::Focused(focused) => {
                        window_focused = *focused;
                    }
                    _ => {}
                }
            }
            Event::RedrawRequested(_) => {
                let now = std::time::Instant::now();
                let dt = now - last_render_time;
                last_render_time = now;
                viewer.update(&mut camera, dt);
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
