use crate::common::{math::face_forward, spectrum::Spectrum};
use crate::pathtracer::TransportMode;
use ambassador::{delegatable_trait, Delegate};

use super::{abs_cos_theta, cos_theta, BxDFInterface, BxDFType};

#[delegatable_trait]
pub trait FresnelInterface {
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
    fn evaluate(&self, _cos_i: f32) -> Spectrum {
        Spectrum::new(1.0)
    }
}

pub struct SpecularReflection {
    r: Spectrum,
    fresnel: Fresnel,
}

impl SpecularReflection {
    pub fn new(r: Spectrum, fresnel: Fresnel) -> Self {
        Self { r, fresnel }
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
        self.fresnel.evaluate(cos_theta(&wi)) * self.r / abs_cos_theta(&wi)
    }

    fn pdf(&self, _wo: &na::Vector3<f32>, _wi: &na::Vector3<f32>) -> f32 {
        0.0
    }
}

pub struct SpecularTransmission {
    t: Spectrum,
    eta_a: f32,
    eta_b: f32,
    fresnel: FresnelDielectric,
    mode: TransportMode,
}

impl SpecularTransmission {
    pub fn new(t: Spectrum, eta_a: f32, eta_b: f32, mode: TransportMode) -> Self {
        Self {
            t,
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
        let mut ft = self.t * (Spectrum::new(1.0) - self.fresnel.evaluate(cos_theta(&wi)));

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
    r: Spectrum,
    t: Spectrum,
    eta_a: f32,
    eta_b: f32,
    mode: TransportMode,
}

impl FresnelSpecular {
    pub fn new(r: Spectrum, t: Spectrum, eta_a: f32, eta_b: f32, mode: TransportMode) -> Self {
        Self {
            r,
            t,
            eta_a,
            eta_b,
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
        u: &na::Point2<f32>,
        pdf: &mut f32,
        sampled_type: &mut Option<BxDFType>,
    ) -> Spectrum {
        let f = fr_dielectric(cos_theta(&wo), self.eta_a, self.eta_b);
        if u[0] < f {
            *wi = na::Vector3::new(-wo.x, -wo.y, wo.z);
            if let Some(sampled_type) = sampled_type {
                *sampled_type = BxDFType::BSDF_REFLECTION | BxDFType::BSDF_SPECULAR;
            }
            *pdf = f;
            f * self.r / abs_cos_theta(&wi)
        } else {
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

            let mut ft = self.t * (Spectrum::new(1.0) - f);

            if self.mode == TransportMode::Radiance {
                ft *= (eta_i * eta_i) / (eta_t * eta_t);
            }

            if let Some(sampled_type) = sampled_type {
                *sampled_type = BxDFType::BSDF_TRANSMISSION | BxDFType::BSDF_SPECULAR;
            }

            *pdf = 1.0 - f;

            ft / abs_cos_theta(&wi)
        }
    }

    fn pdf(&self, _wo: &na::Vector3<f32>, _wi: &na::Vector3<f32>) -> f32 {
        0.0
    }
}
