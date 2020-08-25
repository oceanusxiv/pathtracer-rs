use super::shape::Shape;
use super::{Bounds3, Material, Ray, SurfaceInteraction};
pub trait Primitive {
    fn intersect(&self, r: &Ray) -> Option<SurfaceInteraction>;
    // fn intersect_p(r: &Ray) -> bool;
}

pub struct GeometricPrimitive {
    pub shape: Box<dyn Shape>,
}

impl Primitive for GeometricPrimitive {
    fn intersect(&self, r: &Ray) -> Option<SurfaceInteraction> {
        if let Some((t_hit, isect)) = self.shape.intersect(r) {
            *r.t_max.borrow_mut() = t_hit;
            Some(isect)
        } else {
            None
        }
    }
}

pub struct Aggregate {
    primitives: Vec<Box<dyn Primitive>>,
}

impl Aggregate {
    pub fn new(primitives: Vec<Box<dyn Primitive>>) -> Self {
        Aggregate { primitives }
    }
}

impl Primitive for Aggregate {
    fn intersect(&self, r: &Ray) -> Option<SurfaceInteraction> {
        let mut isect_final: Option<SurfaceInteraction> = None;

        for prim in &self.primitives {
            if let Some(isect) = prim.intersect(r) {
                isect_final = Some(isect);
            }
        }

        isect_final
    }
}
