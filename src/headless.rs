use std::path::PathBuf;

use crate::{
    common::Camera,
    pathtracer::{integrator::PathIntegrator, RenderScene},
};

pub fn run(
    log: slog::Logger,
    render_scene: RenderScene,
    camera: Camera,
    integrator: PathIntegrator,
    output_path: PathBuf,
) {
    integrator.render(&camera, &render_scene);
    camera.film.write_image().save(&output_path).unwrap();
}
