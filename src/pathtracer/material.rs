use super::{SurfaceInteraction, TransportMode};
use crate::common::spectrum::Spectrum;

pub trait BxDF {
    fn f(&self, wo: &glm::Vec3, wi: &glm::Vec3) -> Spectrum;
    fn sample_f(&self, wo: &glm::Vec3) -> (glm::Vec3, f32, Spectrum);
}

const MAX_BXDFS: usize = 8;
pub struct BSDF {
    pub eta: f32,
    ns: na::Vector3<f32>,
    ng: na::Vector3<f32>,
    n_bxdfs: usize,
    bxdfs: [Option<Box<dyn BxDF>>; MAX_BXDFS],
}

impl BSDF {
    // fn new() -> Self {}
}

pub trait Material {
    fn compute_scattering_functions(&self, si: &mut SurfaceInteraction, mode: TransportMode);
}

pub trait SyncMaterial: Material + Send + Sync {}
impl<T> SyncMaterial for T where T: Material + Send + Sync {}

pub struct MatteMaterial {}

impl Material for MatteMaterial {
    fn compute_scattering_functions(&self, si: &mut SurfaceInteraction, mode: TransportMode) {
        todo!()
    }
}
