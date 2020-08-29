use super::shape::{Shape, SyncShape};
use super::{material::SyncMaterial, Material, SurfaceInteraction, TransportMode};
use crate::common::bounds::Bounds3;
use crate::common::ray::Ray;
use std::{rc::Rc, sync::Arc};
pub trait Primitive {
    fn intersect<'a>(&'a self, r: &Ray, isect: &mut SurfaceInteraction<'a>) -> bool;
    fn intersect_p(&self, r: &Ray) -> bool;
    fn world_bound(&self) -> Bounds3;
    fn get_material(&self) -> &dyn SyncMaterial;
    fn compute_scattering_functions(&self, si: &mut SurfaceInteraction, mode: TransportMode);
}

pub trait SyncPrimitive: Primitive + Send + Sync {}
impl<T> SyncPrimitive for T where T: Primitive + Send + Sync {}

pub struct GeometricPrimitive {
    pub shape: Box<dyn SyncShape>,
    pub material: Arc<dyn SyncMaterial>,
}

impl Primitive for GeometricPrimitive {
    fn intersect<'si>(&'si self, r: &Ray, mut isect: &mut SurfaceInteraction<'si>) -> bool {
        let mut t_hit = 0.0f32;
        if !self.shape.intersect(r, &mut t_hit, &mut isect) {
            return false;
        }

        *r.t_max.borrow_mut() = t_hit;
        isect.primitive = Some(self);

        return true;
    }

    fn intersect_p(&self, r: &Ray) -> bool {
        self.shape.intersect_p(r)
    }

    fn world_bound(&self) -> Bounds3 {
        self.shape.world_bound()
    }

    fn get_material(&self) -> &dyn SyncMaterial {
        &*self.material
    }

    fn compute_scattering_functions(&self, si: &mut SurfaceInteraction, mode: TransportMode) {
        self.material.compute_scattering_functions(si, mode);
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
    fn intersect<'a>(&'a self, r: &Ray, mut isect: &mut SurfaceInteraction<'a>) -> bool {
        let mut hit = false;
        for prim in &self.primitives {
            if prim.intersect(r, &mut isect) {
                hit = true;
            }
        }

        hit
    }

    fn intersect_p(&self, r: &Ray) -> bool {
        todo!()
    }

    fn world_bound(&self) -> Bounds3 {
        unimplemented!()
    }

    fn get_material(&self) -> &dyn SyncMaterial {
        unimplemented!()
    }

    fn compute_scattering_functions(&self, si: &mut SurfaceInteraction, mode: TransportMode) {
        unimplemented!()
    }
}
