#[macro_use]
extern crate slog;

use pathtracer_rs::*;
use slog::Drain;

fn main() {
    let decorator = slog_term::TermDecorator::new().build();
    let drain = slog_term::FullFormat::new(decorator).build().fuse();
    let drain = slog_async::Async::new(drain).build().fuse();
    let log = slog::Logger::root(drain.fuse(), o!());
    let scene_path;
    if cfg!(target_os = "windows") {
        scene_path = "C://Users/eric1/Downloads/Sponza/Sponza.gltf";
    } else if cfg!(target_os = "macos") {
        scene_path = "/Users/eric/Downloads/Sponza/Sponza.gltf";
    } else {
        scene_path = "/home/eric/Downloads/Sponza/Sponza.gltf";
    }
    info!(log, "openning scene: {:?}", scene_path);
    let pixel_samples_sqrt = 2;
    let (world, mut camera) = common::World::from_gltf(scene_path, &common::DEFAULT_RESOLUTION);
    let render_scene = pathtracer::RenderScene::from_world(&log, &world);
    let sampler =
        pathtracer::sampling::Sampler::new(pixel_samples_sqrt, pixel_samples_sqrt, true, 8);
    let integrator = pathtracer::integrator::DirectLightingIntegrator::new(&log, sampler, 5);

    integrator.render(&mut camera, &render_scene);
}
