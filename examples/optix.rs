#[macro_use]
extern crate slog;

use pathtracer_rs::*;
use slog::Drain;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let drain = slog::Discard;
    let log = slog::Logger::root(drain.fuse(), o!());
    let scene_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("data/cornell-box.xml");
    let scene_path = scene_path.to_str().unwrap();
    let (mut camera, render_scene, _) =
        common::importer::import(&log, &scene_path, &common::DEFAULT_RESOLUTION, false);

    let mut accel = pathtracer::accelerator::optix::OptixAccelerator::new(&render_scene)?;

    accel.intersect()?;

    Ok(())
}
