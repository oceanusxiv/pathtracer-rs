pub mod bounds;
pub mod film;
pub mod filter;
pub mod importer;
pub mod math;
pub mod ray;
pub mod spectrum;

use film::Film;
use filter::{Filter, GuassianFilter};
use slog::Drain;

lazy_static::lazy_static! {
    pub static ref DEFAULT_RESOLUTION: glm::Vec2 = glm::vec2(640.0, 480.0);
}

static DEFAULT_Z_NEAR: f32 = 0.01;
static DEFAULT_Z_FAR: f32 = 1000.0;

pub struct Camera {
    pub cam_to_world: na::Isometry3<f32>,
    pub cam_to_screen: na::Perspective3<f32>,
    pub screen_to_raster: na::Affine3<f32>,
    pub raster_to_screen: na::Affine3<f32>,

    pub dx_camera: na::Vector3<f32>,
    pub dy_camera: na::Vector3<f32>,

    pub film: Film,
}

impl Camera {
    pub fn new(
        cam_to_world: &na::Isometry3<f32>,
        cam_to_screen: &na::Perspective3<f32>,
        resolution: &glm::Vec2,
    ) -> Camera {
        let screen_to_raster = glm::scaling(&glm::vec3(resolution.x, resolution.y, 1.0))
            * glm::scaling(&glm::vec3(1.0 / (2.0), 1.0 / (-2.0), 1.0))
            * glm::translation(&glm::vec3(1.0, -1.0, 0.0));
        let screen_to_raster = na::Affine3::from_matrix_unchecked(screen_to_raster);
        let resolution = glm::vec2(resolution.x as u32, resolution.y as u32);
        let raster_to_screen = screen_to_raster.inverse();
        let raster_to_camera = cam_to_screen.to_projective().inverse() * raster_to_screen;
        let dx_camera = raster_to_camera * na::Point3::new(1.0, 0.0, 0.0)
            - raster_to_camera * na::Point3::origin();
        let dy_camera = raster_to_camera * na::Point3::new(0.0, 1.0, 0.0)
            - raster_to_camera * na::Point3::origin();

        Self {
            cam_to_world: *cam_to_world,
            cam_to_screen: *cam_to_screen,
            screen_to_raster,
            raster_to_screen,
            dx_camera,
            dy_camera,
            film: Film::new(
                &resolution,
                Box::new(Filter::Guassian(GuassianFilter::new(2.))),
            ),
        }
    }
}

#[derive(Clone, Copy)]
pub enum WrapMode {
    Repeat,
    Black,
    Clamp,
}

pub fn new_drain(
    level: slog::Level,
    allowed_modules: &Option<slog_kvfilter::KVFilterList>,
) -> slog::Fuse<slog::LevelFilter<slog::Fuse<slog_kvfilter::KVFilter<slog::Fuse<slog_async::Async>>>>>
{
    let decorator = slog_term::TermDecorator::new().build();
    let drain = slog_term::CompactFormat::new(decorator).build().fuse();
    let drain = slog_async::Async::new(drain).chan_size(1000).build().fuse();
    let drain = slog_kvfilter::KVFilter::new(drain, slog::Level::Warning)
        .only_pass_any_on_all_keys(allowed_modules.clone())
        .fuse();
    drain.filter_level(level).fuse()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cam_with_look_at(eye: &na::Point3<f32>, center: &na::Point3<f32>) -> Camera {
        Camera::new(
            &na::Isometry3::look_at_rh(eye, center, &glm::vec3(0.0, 1.0, 0.0)).inverse(),
            &na::Perspective3::new(
                DEFAULT_RESOLUTION.x / DEFAULT_RESOLUTION.y,
                std::f32::consts::FRAC_PI_2,
                DEFAULT_Z_NEAR,
                DEFAULT_Z_FAR,
            ),
            &DEFAULT_RESOLUTION,
        )
    }

    #[test]
    fn test_camera_wold_to_screen() {
        let test_cam = cam_with_look_at(&na::Point3::new(10.0, 10.0, 10.0), &na::Point3::origin());

        let test_world_space = na::Point3::origin();
        let test_cam_space = test_cam.cam_to_world.inverse() * test_world_space;
        let test_screen_space = test_cam.cam_to_screen.project_point(&test_cam_space);
        approx::assert_relative_eq!(
            test_cam_space,
            na::Point3::new(0.0, 0.0, -glm::length(&glm::vec3(10.0, 10.0, 10.0))),
            epsilon = 0.000_001
        );

        let z = glm::length(&glm::vec3(10.0, 10.0, 10.0));
        let z_screen =
            ((z - DEFAULT_Z_NEAR) * DEFAULT_Z_FAR) / ((DEFAULT_Z_FAR - DEFAULT_Z_NEAR) * z);

        approx::assert_relative_eq!(
            test_screen_space,
            na::Point3::new(0.0, 0.0, z_screen),
            epsilon = 0.000_001
        );
    }

    #[test]
    fn test_camera_screen_to_raster() {
        let test_cam = cam_with_look_at(&na::Point3::origin(), &na::Point3::new(1.0, 0.0, 0.0));

        let test_screen_space1 = na::Point3::new(1.0, 1.0, 0.5);
        let test_raster_space1 = test_cam.screen_to_raster * test_screen_space1;

        approx::assert_relative_eq!(
            test_raster_space1,
            na::Point3::new(DEFAULT_RESOLUTION.x, DEFAULT_RESOLUTION.y, 0.5),
            epsilon = 0.000_001
        );

        let test_screen_space2 = na::Point3::new(-1.0, -1.0, 0.5);
        let test_raster_space2 = test_cam.screen_to_raster * test_screen_space2;

        approx::assert_relative_eq!(
            test_raster_space2,
            na::Point3::new(0.0, 0.0, 0.5),
            epsilon = 0.000_001
        );
    }

    #[test]
    fn test_camera_raster_to_screen() {
        let test_cam = cam_with_look_at(&na::Point3::origin(), &na::Point3::new(1.0, 0.0, 0.0));

        let test_raster_space1 = na::Point3::new(640.0, 360.0, 0.0);
        let test_cam_space1 = test_cam
            .cam_to_screen
            .unproject_point(&(test_cam.raster_to_screen * test_raster_space1));

        approx::assert_relative_eq!(
            test_cam_space1,
            na::Point3::new(0.0, 0.0, -0.1),
            epsilon = 0.000_001
        );
    }
}
