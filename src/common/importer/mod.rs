use super::Camera;

pub mod gltf;
pub mod mitsuba;

pub fn import(
    log: &slog::Logger,
    path: &str,
    resolution: &na::Vector2<f32>,
) -> (
    Camera,
    crate::pathtracer::RenderScene,
    crate::viewer::ViewerScene,
) {
    let ext = std::path::Path::new(path).extension().unwrap();

    if ext == "gltf" || ext == "glb" {
        gltf::from_gltf(&log, &path, &resolution)
    } else if ext == "xml" {
        mitsuba::from_mitsuba(&log, &path, &resolution)
    } else {
        panic!("unsupported format!");
    }
}
