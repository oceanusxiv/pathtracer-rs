use super::{
    sampling::{cosine_sample_hemisphere, uniform_hemisphere_pdf, uniform_sample_hemisphere},
    TransportMode,
};
use crate::common::{math::face_forward, spectrum::Spectrum};
use ambassador::{delegatable_trait, Delegate};

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

#[delegatable_trait]
pub trait BxDFInterface {
    fn f(&self, wo: &na::Vector3<f32>, wi: &na::Vector3<f32>) -> Spectrum;
    fn sample_f(
        &self,
        wo: &na::Vector3<f32>,
        wi: &mut na::Vector3<f32>,
        u: &na::Point2<f32>,
        pdf: &mut f32,
        _sampled_type: &mut Option<BxDFType>,
    ) -> Spectrum {
        *wi = cosine_sample_hemisphere(&u);
        if wo.z < 0.0 {
            wi.z *= -1.0;
        }

        *pdf = self.pdf(&wo, &wi);
        self.f(&wo, &wi)
    }
    fn rho(
        &self,
        wo: &na::Vector3<f32>,
        n_samples: usize,
        samples: &[na::Point2<f32>],
    ) -> Spectrum {
        let mut r = Spectrum::new(0.0);

        for i in 0..n_samples {
            let mut wi = glm::zero();
            let mut pdf = 0.0;
            let f = self.sample_f(&wo, &mut wi, &samples[i], &mut pdf, &mut None);

            if pdf > 0.0 {
                r += f * abs_cos_theta(&wi) / pdf;
            }
        }

        r / (n_samples as f32)
    }
    fn rho_no_wo(
        &self,
        n_samples: usize,
        samples_1: &[na::Point2<f32>],
        samples_2: &[na::Point2<f32>],
    ) -> Spectrum {
        let mut r = Spectrum::new(0.0);

        for i in 0..n_samples {
            let mut wi = glm::zero();
            let wo = uniform_sample_hemisphere(&samples_1[i]);
            let mut pdf_i = 0.0;
            let pdf_o = uniform_hemisphere_pdf();
            let f = self.sample_f(&wo, &mut wi, &samples_2[i], &mut pdf_i, &mut None);

            if pdf_i > 0.0 {
                r += f * abs_cos_theta(&wi) * abs_cos_theta(&wo) / (pdf_o * pdf_i);
            }
        }

        r / (std::f32::consts::PI * n_samples as f32)
    }

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

#[derive(Delegate)]
#[delegate(BxDFInterface)]
pub enum BxDF {
    Lambertian(LambertianReflection),
    SpecularReflection(SpecularReflection),
    SpecularTransmission(SpecularTransmission),
    FresnelSpecular(FresnelSpecular),
}

pub struct LambertianReflection {
    R: Spectrum,
}

impl LambertianReflection {
    pub fn new(R: Spectrum) -> Self {
        Self { R }
    }
}

impl BxDFInterface for LambertianReflection {
    fn f(&self, _wo: &na::Vector3<f32>, _wi: &na::Vector3<f32>) -> Spectrum {
        self.R * std::f32::consts::FRAC_1_PI
    }

    fn get_type(&self) -> BxDFType {
        BxDFType::BSDF_REFLECTION | BxDFType::BSDF_DIFFUSE
    }

    fn rho(
        &self,
        _wo: &na::Vector3<f32>,
        _n_samples: usize,
        _samples: &[na::Point2<f32>],
    ) -> Spectrum {
        self.R
    }

    fn rho_no_wo(
        &self,
        _n_samples: usize,
        _samples_1: &[na::Point2<f32>],
        _samples_2: &[na::Point2<f32>],
    ) -> Spectrum {
        self.R
    }
}

#[delegatable_trait]
trait FresnelInterface {
    fn evaluate(&self, cos_i: f32) -> Spectrum;
}

#[derive(Delegate)]
#[delegate(FresnelInterface)]
pub enum Fresnel {
    Dielectric(FresnelDielectric),
    Conductor(FresnelConductor),
    NoOp(FresnelNoOp),
}

fn fr_dielectric(cos_theta_i: f32, mut eta_i: f32, mut eta_t: f32) -> f32 {
    let mut cos_theta_i = cos_theta_i.clamp(-1.0, 1.0);
    let entering = cos_theta_i > 0.0;
    if !entering {
        std::mem::swap(&mut eta_i, &mut eta_t);
        cos_theta_i = cos_theta_i.abs();
    }

    let sin_theta_i = 0.0f32.max(1.0 - cos_theta_i * cos_theta_i).sqrt();
    let sin_theta_t = eta_i / eta_t * sin_theta_i;
    if sin_theta_t >= 1.0 {
        return 1.0;
    }
    let cos_theta_t = 0.0f32.max(1.0 - sin_theta_t * sin_theta_t).sqrt();
    let r_parl = ((eta_t * cos_theta_i) - (eta_i * cos_theta_t))
        / ((eta_t * cos_theta_i) + (eta_i * cos_theta_t));
    let r_perp = ((eta_i * cos_theta_i) - (eta_t * cos_theta_t))
        / ((eta_i * cos_theta_i) + (eta_t * cos_theta_t));
    return (r_parl * r_parl + r_perp * r_perp) / 2.0;
}

pub struct FresnelDielectric {
    eta_i: f32,
    eta_t: f32,
}

impl FresnelDielectric {
    pub fn new(eta_i: f32, eta_t: f32) -> Self {
        Self { eta_i, eta_t }
    }
}

impl FresnelInterface for FresnelDielectric {
    fn evaluate(&self, cos_i: f32) -> Spectrum {
        Spectrum::new(fr_dielectric(cos_i, self.eta_i, self.eta_t))
    }
}

pub struct FresnelConductor {}

impl FresnelInterface for FresnelConductor {
    fn evaluate(&self, cos_i: f32) -> Spectrum {
        todo!()
    }
}

pub struct FresnelNoOp {}

impl FresnelInterface for FresnelNoOp {
    fn evaluate(&self, cos_i: f32) -> Spectrum {
        Spectrum::new(1.0)
    }
}

pub struct SpecularReflection {
    R: Spectrum,
    fresnel: Fresnel,
}

impl SpecularReflection {
    pub fn new(R: Spectrum, fresnel: Fresnel) -> Self {
        Self { R, fresnel }
    }
}

impl BxDFInterface for SpecularReflection {
    fn f(&self, _wo: &na::Vector3<f32>, _wi: &na::Vector3<f32>) -> Spectrum {
        Spectrum::new(0.0)
    }

    fn get_type(&self) -> BxDFType {
        BxDFType::BSDF_REFLECTION | BxDFType::BSDF_SPECULAR
    }

    fn sample_f(
        &self,
        wo: &na::Vector3<f32>,
        wi: &mut na::Vector3<f32>,
        _u: &na::Point2<f32>,
        pdf: &mut f32,
        _sampled_type: &mut Option<BxDFType>,
    ) -> Spectrum {
        *wi = na::Vector3::new(-wo.x, -wo.y, wo.z);
        *pdf = 1.0;
        self.fresnel.evaluate(cos_theta(&wi)) * self.R / abs_cos_theta(&wi)
    }

    fn pdf(&self, _wo: &na::Vector3<f32>, _wi: &na::Vector3<f32>) -> f32 {
        0.0
    }
}

pub struct SpecularTransmission {
    T: Spectrum,
    eta_a: f32,
    eta_b: f32,
    fresnel: FresnelDielectric,
    mode: TransportMode,
}

impl SpecularTransmission {
    pub fn new(T: Spectrum, eta_a: f32, eta_b: f32, mode: TransportMode) -> Self {
        Self {
            T,
            eta_a,
            eta_b,
            fresnel: FresnelDielectric {
                eta_i: eta_a,
                eta_t: eta_b,
            },
            mode,
        }
    }
}

fn refract(
    wi: &na::Vector3<f32>,
    n: &na::Vector3<f32>,
    eta: f32,
    wt: &mut na::Vector3<f32>,
) -> bool {
    let cos_theta_i = n.dot(&wi);
    let sin_2_theta_i = 0.0f32.max(1.0 - cos_theta_i * cos_theta_i);
    let sin_2_theta_t = eta * eta * sin_2_theta_i;
    if sin_2_theta_t > 1.0 {
        return false;
    }
    let cos_theta_t = (1.0 - sin_2_theta_t).sqrt();
    *wt = eta * -wi + (eta * cos_theta_i - cos_theta_t) * n;

    true
}

impl BxDFInterface for SpecularTransmission {
    fn f(&self, _wo: &na::Vector3<f32>, _wi: &na::Vector3<f32>) -> Spectrum {
        Spectrum::new(0.0)
    }

    fn get_type(&self) -> BxDFType {
        BxDFType::BSDF_TRANSMISSION | BxDFType::BSDF_SPECULAR
    }

    fn sample_f(
        &self,
        wo: &na::Vector3<f32>,
        mut wi: &mut na::Vector3<f32>,
        _u: &na::Point2<f32>,
        pdf: &mut f32,
        _sampled_type: &mut Option<BxDFType>,
    ) -> Spectrum {
        let entering = cos_theta(&wo) > 0.0;
        let eta_i = if entering { self.eta_a } else { self.eta_b };
        let eta_t = if entering { self.eta_b } else { self.eta_a };

        if !refract(
            &wo,
            &face_forward(&na::Vector3::new(0.0, 0.0, 1.0), &wo),
            eta_i / eta_t,
            &mut wi,
        ) {
            return Spectrum::new(0.0);
        }

        *pdf = 1.0;
        let mut ft = self.T * (Spectrum::new(1.0) - self.fresnel.evaluate(cos_theta(&wi)));

        if self.mode == TransportMode::Radiance {
            ft *= (eta_i * eta_i) / (eta_t * eta_t);
        }

        ft / abs_cos_theta(&wi)
    }

    fn pdf(&self, _wo: &na::Vector3<f32>, _wi: &na::Vector3<f32>) -> f32 {
        0.0
    }
}

pub struct FresnelSpecular {
    R: Spectrum,
    T: Spectrum,
    eta_a: f32,
    eta_b: f32,
    fresnel: FresnelDielectric,
    mode: TransportMode,
}

impl FresnelSpecular {
    pub fn new(R: Spectrum, T: Spectrum, eta_a: f32, eta_b: f32, mode: TransportMode) -> Self {
        Self {
            R,
            T,
            eta_a,
            eta_b,
            fresnel: FresnelDielectric {
                eta_i: eta_a,
                eta_t: eta_b,
            },
            mode,
        }
    }
}

impl BxDFInterface for FresnelSpecular {
    fn f(&self, _wo: &na::Vector3<f32>, _wi: &na::Vector3<f32>) -> Spectrum {
        Spectrum::new(0.0)
    }

    fn get_type(&self) -> BxDFType {
        BxDFType::BSDF_REFLECTION | BxDFType::BSDF_TRANSMISSION | BxDFType::BSDF_SPECULAR
    }

    fn sample_f(
        &self,
        wo: &na::Vector3<f32>,
        mut wi: &mut na::Vector3<f32>,
        _u: &na::Point2<f32>,
        pdf: &mut f32,
        _sampled_type: &mut Option<BxDFType>,
    ) -> Spectrum {
        todo!()
    }

    fn pdf(&self, _wo: &na::Vector3<f32>, _wi: &na::Vector3<f32>) -> f32 {
        0.0
    }
}
