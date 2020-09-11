#[macro_use]
extern crate slog;

use pathtracer_rs::*;
use slog::Drain;

fn main() {
    let drain = slog::Discard;
    let log = slog::Logger::root(drain.fuse(), o!());
    let scene_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("data/cornell-box.xml");
    let scene_path = scene_path.to_str().unwrap();
    let pixel_samples_sqrt = 5;
    let (mut camera, render_scene, _) =
        common::importer::import(&log, &scene_path, &common::DEFAULT_RESOLUTION, false);
    let sampler = pathtracer::sampling::SamplerBuilder::new(
        &log,
        pixel_samples_sqrt,
        pixel_samples_sqrt,
        true,
        8,
    );
    let mut integrator = pathtracer::integrator::PathIntegrator::new(&log, sampler, 5);
    integrator.preprocess(&render_scene);
    integrator.render(&mut camera, &render_scene);
}
