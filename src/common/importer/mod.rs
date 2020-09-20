use super::Camera;

pub mod gltf;
pub mod mitsuba;

pub fn import(
    log: &slog::Logger,
    path: &str,
    resolution: &na::Vector2<f32>,
    default_lights: bool,
) -> (
    Camera,
    crate::pathtracer::RenderScene,
    crate::viewer::renderer::ViewerScene,
) {
    let ext = std::path::Path::new(path).extension().unwrap();

    if ext == "gltf" || ext == "glb" {
        gltf::from_gltf(&log, &path, &resolution, default_lights)
    } else if ext == "xml" {
        mitsuba::from_mitsuba(&log, &path, &resolution)
    } else {
        panic!("unsupported format!");
    }
}
