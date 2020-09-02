use super::sampling::cosine_sample_hemisphere;
use crate::common::spectrum::Spectrum;

fn cos_theta(w: &na::Vector3<f32>) -> f32 {
    w.z
}

fn cos_2_theta(w: &na::Vector3<f32>) -> f32 {
    w.z * w.z
}

fn abs_cos_theta(w: &na::Vector3<f32>) -> f32 {
    w.z.abs()
}

fn same_hemisphere(w: &na::Vector3<f32>, wp: &na::Vector3<f32>) -> bool {
    w.z * wp.z > 0.0
}

bitflags! {
    pub struct BxDFType: u32 {
        const BSDF_REFLECTION = 1 << 0;
        const BSDF_TRANSMISSION = 1 << 1;
        const BSDF_DIFFUSE = 1 << 2;
        const BSDF_GLOSSY = 1 << 3;
        const BSDF_SPECULAR = 1 << 4;
        const BSDF_ALL = Self::BSDF_DIFFUSE.bits | Self::BSDF_GLOSSY.bits | Self::BSDF_SPECULAR.bits | Self::BSDF_REFLECTION.bits |
        Self::BSDF_TRANSMISSION.bits;
    }
}
pub trait BxDF {
    fn f(&self, wo: &na::Vector3<f32>, wi: &na::Vector3<f32>) -> Spectrum;
    fn sample_f(
        &self,
        wo: &na::Vector3<f32>,
        wi: &mut na::Vector3<f32>,
        u: &na::Point2<f32>,
        pdf: &mut f32,
        sampled_type: &mut Option<BxDFType>,
    ) -> Spectrum {
        *wi = cosine_sample_hemisphere(&u);
        if wo.z < 0.0 {
            wi.z *= -1.0;
        }

        *pdf = self.pdf(&wo, &wi);
        self.f(&wo, &wi)
    }
    fn rho(&self, wo: &na::Vector3<f32>, n_samples: usize, samples: &na::Point2<f32>) -> Spectrum;
    fn rho_no_wo(
        &self,
        n_samples: usize,
        samples_1: &na::Point2<f32>,
        samples_2: &na::Point2<f32>,
    ) -> Spectrum;

    fn matches_flags(&self, t: BxDFType) -> bool {
        (self.get_type() & t) == self.get_type()
    }
    fn get_type(&self) -> BxDFType;
    fn pdf(&self, wo: &na::Vector3<f32>, wi: &na::Vector3<f32>) -> f32 {
        if same_hemisphere(&wo, &wi) {
            abs_cos_theta(&wi) * std::f32::consts::FRAC_1_PI
        } else {
            0.0
        }
    }
}

pub struct LambertianReflection {
    R: Spectrum,
    pub bxdf_type: BxDFType,
}

impl LambertianReflection {
    pub fn new(R: Spectrum) -> Self {
        LambertianReflection {
            R,
            bxdf_type: BxDFType::BSDF_REFLECTION | BxDFType::BSDF_DIFFUSE,
        }
    }
}

impl BxDF for LambertianReflection {
    fn f(&self, wo: &na::Vector3<f32>, wi: &na::Vector3<f32>) -> Spectrum {
        self.R * std::f32::consts::FRAC_1_PI
    }

    fn get_type(&self) -> BxDFType {
        self.bxdf_type
    }

    fn rho(&self, wo: &na::Vector3<f32>, n_samples: usize, samples: &na::Point2<f32>) -> Spectrum {
        self.R
    }

    fn rho_no_wo(
        &self,
        n_samples: usize,
        samples_1: &na::Point2<f32>,
        samples_2: &na::Point2<f32>,
    ) -> Spectrum {
        self.R
    }
}
