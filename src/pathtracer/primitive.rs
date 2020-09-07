use super::shape::SyncShape;
use super::{
    light::DiffuseAreaLight, Material, MaterialInterface, SurfaceInteraction, TransportMode,
};
use crate::common::bounds::Bounds3;
use crate::common::ray::Ray;
use std::sync::Arc;
pub trait Primitive {
    fn intersect<'a>(&'a self, r: &Ray, isect: &mut SurfaceInteraction<'a>) -> bool;
    fn intersect_p(&self, r: &Ray) -> bool;
    fn world_bound(&self) -> Bounds3;
    fn get_material(&self) -> &Material;
    fn compute_scattering_functions(&self, si: &mut SurfaceInteraction, mode: TransportMode);
    fn get_area_light(&self) -> Option<&DiffuseAreaLight>;
}

pub trait SyncPrimitive: Primitive + Send + Sync {}
impl<T> SyncPrimitive for T where T: Primitive + Send + Sync {}

pub struct GeometricPrimitive {
    shape: Arc<dyn SyncShape>,
    material: Arc<Material>,
    area_light: Option<Arc<DiffuseAreaLight>>,
}

impl GeometricPrimitive {
    pub fn new(
        shape: Arc<dyn SyncShape>,
        material: Arc<Material>,
        area_light: Option<Arc<DiffuseAreaLight>>,
    ) -> Self {
        Self {
            shape,
            material,
            area_light,
        }
    }
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

    fn get_material(&self) -> &Material {
        self.material.as_ref()
    }

    fn compute_scattering_functions(&self, si: &mut SurfaceInteraction, mode: TransportMode) {
        self.material.compute_scattering_functions(si, mode);
    }

    fn get_area_light(&self) -> Option<&DiffuseAreaLight> {
        self.area_light.as_deref()
    }
}
