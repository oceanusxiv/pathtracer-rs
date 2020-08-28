use super::SurfaceInteraction;
use super::spectrum::Spectrum;

pub trait BxDF {
    fn f(&self, wo: &glm::Vec3, wi: &glm::Vec3) -> Spectrum;
    fn sample_f(&self, wo: &glm::Vec3) -> (glm::Vec3, f32, Spectrum);
}

pub trait BSDF {}

pub trait Material {
    fn compute_scattering_functions(&self, si: &mut SurfaceInteraction) -> Box<dyn BSDF>;
}
