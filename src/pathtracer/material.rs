use super::{
    bsdf::BSDF,
    bxdf::{BxDF, LambertianReflection},
    SurfaceInteraction, TransportMode,
};
use crate::common::spectrum::Spectrum;
use ambassador::{delegatable_trait, Delegate};

#[delegatable_trait]
pub trait MaterialInterface {
    fn compute_scattering_functions(&self, si: &mut SurfaceInteraction, mode: TransportMode);
}

#[derive(Delegate)]
#[delegate(MaterialInterface)]
pub enum Material {
    Matte(MatteMaterial),
}

pub struct MatteMaterial {}

impl MaterialInterface for MatteMaterial {
    fn compute_scattering_functions(&self, si: &mut SurfaceInteraction, mode: TransportMode) {
        let mut bsdf = BSDF::new(&si, 1.0);
        let r = Spectrum::new(1.0);
        bsdf.add(BxDF::Lambertian(LambertianReflection::new(r)));

        si.bsdf = Some(bsdf);
    }
}
