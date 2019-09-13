mod material;
mod sampling;

use crate::common::{Bounds3, Camera};
use material::Material;
use std::rc::Rc;

pub struct Ray {
    pub o: glm::Vec3,
    pub d: glm::Vec3,
    pub t: f32,
}

impl Ray {
    pub fn point_at(&self, t: f32) -> glm::Vec3 {
        self.o + self.d * t
    }
}

pub struct SurfaceInteraction {
    pub p: glm::Vec3,
    pub time: f32,
    pub p_error: glm::Vec3,
    pub wo: glm::Vec3,
    pub n: glm::Vec3,
}

pub trait Primitive {
    fn intersect(r: &Ray) -> Option<SurfaceInteraction>;
    fn intersect_p(r: &Ray) -> bool;
    fn get_material() -> Rc<dyn Material>;
    fn world_bound() -> Bounds3;
}

pub trait Sampler {}

impl Camera {
    pub fn generate_ray(&self, film_point: glm::Vec2) -> Ray {
        let cam_dir = self.raster_to_cam * glm::vec4(film_point.x, film_point.y, 0.0, 1.0);
        let cam_orig = glm::vec4(0.0, 0.0, 0.0, 1.0);
        let world_orig = self.cam_to_world * cam_orig;
        let world_dir = self.cam_to_world * cam_dir;
        Ray {
            o: world_orig.xyz() / world_orig.w,
            d: world_dir.xyz() / world_dir.w,
            t: std::f32::INFINITY,
        }
    }
}
