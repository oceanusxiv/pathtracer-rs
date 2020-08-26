use super::shape::Shape;
use super::{Bounds3, Material, Ray, SurfaceInteraction};
use std::rc::Rc;
pub trait Primitive {
    fn intersect(&self, r: &Ray) -> Option<SurfaceInteraction>;
    // fn intersect_p(r: &Ray) -> bool;
    fn world_bound(&self) -> Bounds3;
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
    fn intersect(&self, r: &Ray) -> Option<SurfaceInteraction> {
        let mut isect_final: Option<SurfaceInteraction> = None;

        for prim in &self.primitives {
            if let Some(isect) = prim.intersect(r) {
                isect_final = Some(isect);
            }
        }

        isect_final
    }

    fn world_bound(&self) -> Bounds3 {
        unimplemented!()
    }
}
