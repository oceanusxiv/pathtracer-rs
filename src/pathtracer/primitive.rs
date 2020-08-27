use super::shape::Shape;
use super::{Bounds3, Material, Ray, SurfaceInteraction};
use std::rc::Rc;
pub trait Primitive {
    fn intersect(&self, r: &Ray, isect: &mut SurfaceInteraction) -> bool;
    // fn intersect_p(r: &Ray) -> bool;
    fn world_bound(&self) -> Bounds3;
}

pub struct GeometricPrimitive {
    pub shape: Box<dyn Shape>,
}

impl Primitive for GeometricPrimitive {
    fn intersect(&self, r: &Ray, mut isect: &mut SurfaceInteraction) -> bool {
        let mut t_hit = 0.0f32;
        if !self.shape.intersect(r, &mut t_hit, &mut isect) {
            return false;
        }

        *r.t_max.borrow_mut() = t_hit;

        return true;
    }

    fn world_bound(&self) -> Bounds3 {
        self.shape.world_bound()
    }
}

pub struct Aggregate {
    primitives: Vec<Rc<dyn Primitive>>,
}

impl Aggregate {
    pub fn new(primitives: Vec<Rc<dyn Primitive>>) -> Self {
        Aggregate { primitives }
    }
}

impl Primitive for Aggregate {
    fn intersect(&self, r: &Ray, mut isect: &mut SurfaceInteraction) -> bool {
        let mut hit = false;
        for prim in &self.primitives {
            if prim.intersect(r, &mut isect) {
                hit = true;
            }
        }

        hit
    }

    fn world_bound(&self) -> Bounds3 {
        unimplemented!()
    }
}
