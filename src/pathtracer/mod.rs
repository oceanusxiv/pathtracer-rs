pub mod accelerator;
mod bsdf;
mod bxdf;
pub mod importer;
pub mod integrator;
mod interaction;
pub mod light;
mod material;
mod primitive;
pub mod sampler;
pub mod sampling;
mod shape;
mod texture;

use crate::common::ray::{Ray, RayDifferential};

use crate::common::Camera;
use interaction::SurfaceMediumInteraction;
use light::SyncLight;
use material::{Material, MaterialInterface};
use primitive::SyncPrimitive;
use std::cell::RefCell;
use std::sync::Arc;

#[derive(PartialEq, Eq)]
pub enum TransportMode {
    Radiance,
    Importance,
}

#[derive(Debug)]
pub struct CameraSample {
    p_film: na::Point2<f32>,
}

impl Camera {
    pub fn generate_ray(&self, sample: &CameraSample) -> Ray {
        let p_camera = self.cam_to_screen.unproject_point(
            &(self.raster_to_screen * na::Point3::new(sample.p_film.x, sample.p_film.y, 0.0)),
        );

        let cam_orig = na::Point3::<f32>::new(0.0, 0.0, 0.0);
        let world_orig = self.cam_to_world * cam_orig;
        let world_dir = self.cam_to_world * p_camera.coords;
        Ray {
            o: world_orig,
            d: world_dir.normalize(),
            t_max: RefCell::new(f32::INFINITY),
        }
    }

    pub fn generate_ray_differential(&self, sample: &CameraSample) -> RayDifferential {
        let p_camera = self.cam_to_screen.unproject_point(
            &(self.raster_to_screen * na::Point3::new(sample.p_film.x, sample.p_film.y, 0.0)),
        );

        let cam_orig = na::Point3::<f32>::new(0.0, 0.0, 0.0);
        let world_orig = self.cam_to_world * cam_orig;
        let world_dir = self.cam_to_world * p_camera.coords;
        let rx_world_dir = self.cam_to_world * (p_camera.coords + self.dx_camera);
        let ry_world_dir = self.cam_to_world * (p_camera.coords + self.dy_camera);
        RayDifferential {
            ray: Ray {
                o: world_orig,
                d: world_dir.normalize(),
                t_max: RefCell::new(f32::INFINITY),
            },
            has_differentials: true,
            rx_origin: world_orig,
            ry_origin: world_orig,
            rx_direction: rx_world_dir.normalize(),
            ry_direction: ry_world_dir.normalize(),
        }
    }
}

pub struct RenderScene {
    scene: Box<dyn SyncPrimitive>,
    pub lights: Vec<Arc<dyn SyncLight>>,
    pub infinite_lights: Vec<Arc<dyn SyncLight>>,
}

impl RenderScene {
    pub fn intersect<'a>(&'a self, r: &Ray, isect: &mut SurfaceMediumInteraction<'a>) -> bool {
        self.scene.intersect(r, isect)
    }

    pub fn intersect_p(&self, r: &Ray) -> bool {
        self.scene.intersect_p(r)
    }
}
