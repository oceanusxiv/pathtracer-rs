use super::{bsdf::BSDF, bxdf::LambertianReflection, SurfaceInteraction, TransportMode};
use crate::common::spectrum::Spectrum;

pub trait Material {
    fn compute_scattering_functions(&self, si: &mut SurfaceInteraction, mode: TransportMode);
}

pub trait SyncMaterial: Material + Send + Sync {}
impl<T> SyncMaterial for T where T: Material + Send + Sync {}

pub struct MatteMaterial {}

impl Material for MatteMaterial {
    fn compute_scattering_functions(&self, si: &mut SurfaceInteraction, mode: TransportMode) {
        let mut bsdf = BSDF::new(&si, 1.0);
        let r = Spectrum::new(1.0);
        bsdf.add(Box::new(LambertianReflection::new(r)));

        si.bsdf = Some(bsdf);
    }
}
