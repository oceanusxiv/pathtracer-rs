use crate::{common::spectrum::Spectrum, pathtracer::TransportMode};

use super::{
    abs_cos_theta, cos_2_phi, cos_2_theta, cos_theta,
    fresnel::FresnelDielectric,
    fresnel::{Fresnel, FresnelInterface},
    same_hemisphere, sin_2_phi, tan_2_theta, tan_theta, BxDFInterface, BxDFType,
};

pub struct TrowbridgeReitzDistribution {
    alpha_x: f32,
    alpha_y: f32,
}

impl TrowbridgeReitzDistribution {
    pub fn d(&self, wh: &na::Vector3<f32>) -> f32 {
        let tan_2_theta = tan_2_theta(&wh);
        if tan_2_theta.is_infinite() {
            return 0.0;
        }
        let cos_4_theta = cos_2_theta(&wh) * cos_2_theta(&wh);
        let e = (cos_2_phi(&wh) / (self.alpha_x * self.alpha_x)
            + sin_2_phi(&wh) / (self.alpha_y * self.alpha_y))
            * tan_2_theta;
        1.0 / (std::f32::consts::PI
            * self.alpha_x
            * self.alpha_y
            * cos_4_theta
            * (1.0 + e)
            * (1.0 + e))
    }

    fn lambda(&self, w: &na::Vector3<f32>) -> f32 {
        let abs_tan_theta = tan_theta(&w).abs();
        if abs_tan_theta.is_infinite() {
            return 0.0;
        }
        let alpha = ((cos_2_phi(&w) * self.alpha_x * self.alpha_x)
            + (sin_2_phi(&w) * self.alpha_y * self.alpha_y))
            .sqrt();
        let alpha_2_tan_2_theta = (alpha * abs_tan_theta) * (alpha * abs_tan_theta);
        (-1.0 + (1.0 + alpha_2_tan_2_theta).sqrt()) / 2.0
    }

    pub fn g1(&self, w: &na::Vector3<f32>) -> f32 {
        1.0 / (1.0 + self.lambda(&w))
    }

    pub fn g(&self, wo: &na::Vector3<f32>, wi: &na::Vector3<f32>) -> f32 {
        1.0 / (1.0 + self.lambda(&wo) + self.lambda(&wi))
    }

    pub fn sample_wh(&self, wo: &na::Vector3<f32>, u: &na::Point2<f32>) -> na::Vector3<f32> {
        todo!()
    }

    pub fn pdf(&self, wo: &na::Vector3<f32>, wh: &na::Vector3<f32>) -> f32 {
        todo!()
    }

    pub fn roughness_to_alpha(roughness: f32) -> f32 {
        let roughness = roughness.max(1e-3);
        let x = roughness.ln();
        1.62142
            + 0.819955 * x
            + 0.1734 * x * x
            + 0.0171201 * x * x * x
            + 0.000640711 * x * x * x * x
    }
}

pub type MicrofacetDistribution = TrowbridgeReitzDistribution;

struct MicrofacetReflection {
    r: Spectrum,
    distribution: Box<MicrofacetDistribution>,
    fresnel: Box<Fresnel>,
}

impl BxDFInterface for MicrofacetReflection {
    fn f(&self, wo: &na::Vector3<f32>, wi: &na::Vector3<f32>) -> Spectrum {
        let cos_theta_o = abs_cos_theta(&wo);
        let cos_theta_i = abs_cos_theta(&wi);
        let wh = wi + wo;
        if cos_theta_i == 0. || cos_theta_o == 0. {
            return Spectrum::new(0.);
        }
        if wh.x == 0. && wh.y == 0. && wh.z == 0. {
            return Spectrum::new(0.);
        }

        let wh = wh.normalize();
        let f = self.fresnel.evaluate(wi.dot(&wh));
        self.r * self.distribution.d(&wh) * self.distribution.g(&wo, &wi) * f
            / (4.0 * cos_theta_i * cos_theta_o)
    }

    fn get_type(&self) -> BxDFType {
        BxDFType::BSDF_REFLECTION | BxDFType::BSDF_GLOSSY
    }

    fn sample_f(
        &self,
        wo: &na::Vector3<f32>,
        wi: &mut na::Vector3<f32>,
        u: &na::Point2<f32>,
        pdf: &mut f32,
        sampled_type: &mut Option<BxDFType>,
    ) -> Spectrum {
        todo!()
    }

    fn pdf(&self, wo: &na::Vector3<f32>, wi: &na::Vector3<f32>) -> f32 {
        todo!()
    }
}

struct MicrofacetTransmission {
    t: Spectrum,
    distribution: Box<MicrofacetDistribution>,
    eta_a: f32,
    eta_b: f32,
    fresnel: FresnelDielectric,
    mode: TransportMode,
}

impl BxDFInterface for MicrofacetTransmission {
    fn f(&self, wo: &na::Vector3<f32>, wi: &na::Vector3<f32>) -> Spectrum {
        if same_hemisphere(&wo, &wi) {
            return Spectrum::new(0.);
        }

        let cos_theta_o = abs_cos_theta(&wo);
        let cos_theta_i = abs_cos_theta(&wi);
        if cos_theta_i == 0. || cos_theta_o == 0. {
            return Spectrum::new(0.);
        }

        let eta = if cos_theta(&wo) > 0.0 {
            self.eta_b / self.eta_a
        } else {
            self.eta_a / self.eta_b
        };
        let mut wh = (wo + wi * eta).normalize();
        if wh.z < 0.0 {
            wh = -wh;
        }

        if wo.dot(&wh) * wi.dot(&wh) > 0. {
            return Spectrum::new(0.);
        }

        let f = self.fresnel.evaluate(wo.dot(&wh));

        let sqrt_denom = wo.dot(&wh) + eta * wi.dot(&wh);
        let factor = if self.mode == TransportMode::Radiance {
            1.0 / eta
        } else {
            1.0
        };

        (Spectrum::new(1.) - f)
            * self.t
            * (self.distribution.d(&wh)
                * self.distribution.g(&wo, &wi)
                * eta
                * eta
                * wi.dot(&wh).abs()
                * wo.dot(&wh).abs()
                * factor
                * factor
                / (cos_theta_i * cos_theta_o * sqrt_denom * sqrt_denom))
    }

    fn get_type(&self) -> BxDFType {
        BxDFType::BSDF_TRANSMISSION | BxDFType::BSDF_GLOSSY
    }

    fn sample_f(
        &self,
        wo: &na::Vector3<f32>,
        wi: &mut na::Vector3<f32>,
        u: &na::Point2<f32>,
        pdf: &mut f32,
        sampled_type: &mut Option<BxDFType>,
    ) -> Spectrum {
        todo!()
    }

    fn pdf(&self, wo: &na::Vector3<f32>, wi: &na::Vector3<f32>) -> f32 {
        todo!()
    }
}

pub struct FresnelBlend {
    rd: Spectrum,
    rs: Spectrum,
    distribution: Box<MicrofacetDistribution>,
}

impl FresnelBlend {
    pub fn schlick_fresnel(&self, cos_theta: f32) -> Spectrum {
        let pow5 = |v: f32| (v * v) * (v * v) * v;
        self.rs + pow5(1.0 - cos_theta) * (Spectrum::new(1.) - self.rs)
    }
}

impl BxDFInterface for FresnelBlend {
    fn f(&self, _wo: &na::Vector3<f32>, _wi: &na::Vector3<f32>) -> Spectrum {
        let pow5 = |v: f32| (v * v) * (v * v) * v;
        todo!()
    }

    fn get_type(&self) -> BxDFType {
        BxDFType::BSDF_REFLECTION | BxDFType::BSDF_SPECULAR
    }

    fn sample_f(
        &self,
        wo: &na::Vector3<f32>,
        mut wi: &mut na::Vector3<f32>,
        u: &na::Point2<f32>,
        pdf: &mut f32,
        sampled_type: &mut Option<BxDFType>,
    ) -> Spectrum {
        todo!()
    }

    fn pdf(&self, _wo: &na::Vector3<f32>, _wi: &na::Vector3<f32>) -> f32 {
        todo!()
    }
}
