use crate::{common::spectrum::Spectrum, pathtracer::TransportMode};

use super::{
    abs_cos_theta, cos_2_phi, cos_2_theta, cos_phi, cos_theta,
    fresnel::FresnelDielectric,
    fresnel::{Fresnel, FresnelInterface},
    reflect, refract, same_hemisphere, sin_2_phi, sin_phi, tan_2_theta, tan_theta, BxDFInterface,
    BxDFType,
};

pub trait MicrofacetDistribution {
    fn d(&self, wh: &na::Vector3<f32>) -> f32;

    fn lambda(&self, w: &na::Vector3<f32>) -> f32;

    fn g1(&self, w: &na::Vector3<f32>) -> f32 {
        1.0 / (1.0 + self.lambda(&w))
    }

    fn g(&self, wo: &na::Vector3<f32>, wi: &na::Vector3<f32>) -> f32 {
        1.0 / (1.0 + self.lambda(&wo) + self.lambda(&wi))
    }

    fn sample_wh(&self, wo: &na::Vector3<f32>, u: &na::Point2<f32>) -> na::Vector3<f32>;

    fn pdf(&self, wo: &na::Vector3<f32>, wh: &na::Vector3<f32>) -> f32;
}

fn trowbridge_reitz_sample_11(
    cos_theta: f32,
    u1: f32,
    mut u2: f32,
    slope_x: &mut f32,
    slope_y: &mut f32,
) {
    if cos_theta > 0.9999 {
        let r = (u1 / (1. - u1)).sqrt();
        let phi = 6.28318530718 * u2;
        *slope_x = r * phi.cos();
        *slope_y = r * phi.sin();
        return;
    }

    let sin_theta = 0.0f32.max(1. - cos_theta * cos_theta).sqrt();
    let tan_theta = sin_theta / cos_theta;
    let alpha = 1. / tan_theta;
    let g1 = 2. / (1. + (1. + 1. / (alpha * alpha)).sqrt());

    let a = 2. * u1 / g1 - 1.;
    let mut tmp = 1. / (a * a - 1.);
    if tmp > 1e10 {
        tmp = 1e10;
    }
    let b = tan_theta;
    let d = 0.0f32.max(b * b * tmp * tmp - (a * a - b * b) * tmp).sqrt();
    let slope_x_1 = b * tmp - d;
    let slope_x_2 = b * tmp + d;
    *slope_x = if a < 0. || slope_x_2 > (1. / tan_theta) {
        slope_x_1
    } else {
        slope_x_2
    };

    let s;
    if u2 > 0.5 {
        s = 1.;
        u2 = 2. * (u2 - 0.5);
    } else {
        s = -1.;
        u2 = 2. * (0.5 - u2);
    }
    let z = (u2 * (u2 * (u2 * 0.27385 - 0.73369) + 0.46341))
        / (u2 * (u2 * (u2 * 0.093073 + 0.309420) - 1.000000) + 0.597999);
    *slope_y = s * z * (1. + *slope_x * *slope_x).sqrt();

    debug_assert!(!slope_y.is_infinite());
    debug_assert!(!slope_y.is_nan());
}

fn trowbridge_reitz_sample(
    wi: &na::Vector3<f32>,
    alpha_x: f32,
    alpha_y: f32,
    u1: f32,
    u2: f32,
) -> na::Vector3<f32> {
    let wi_stretched = na::Vector3::new(alpha_x * wi.x, alpha_y * wi.y, wi.z).normalize();

    let mut slope_x = 0.0;
    let mut slope_y = 0.0;
    trowbridge_reitz_sample_11(cos_theta(&wi_stretched), u1, u2, &mut slope_x, &mut slope_y);

    let tmp = cos_phi(&wi_stretched) * slope_x - sin_phi(&wi_stretched) * slope_y;
    slope_y = sin_phi(&wi_stretched) * slope_x + cos_phi(&wi_stretched) * slope_y;
    slope_x = tmp;

    slope_x = alpha_x * slope_x;
    slope_y = alpha_y * slope_y;

    na::Vector3::new(-slope_x, -slope_y, 1.).normalize()
}

pub struct TrowbridgeReitzDistribution {
    alpha_x: f32,
    alpha_y: f32,
}

impl TrowbridgeReitzDistribution {
    pub fn new(alpha_x: f32, alpha_y: f32) -> Self {
        Self {
            alpha_x: alpha_x.max(0.001),
            alpha_y: alpha_y.max(0.001),
        }
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

impl MicrofacetDistribution for TrowbridgeReitzDistribution {
    fn d(&self, wh: &na::Vector3<f32>) -> f32 {
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

    fn sample_wh(&self, wo: &na::Vector3<f32>, u: &na::Point2<f32>) -> na::Vector3<f32> {
        let flip = wo.z < 0.;
        let wo = if flip { -wo } else { *wo };
        let wh = trowbridge_reitz_sample(&wo, self.alpha_x, self.alpha_y, u[0], u[1]);
        if flip {
            -wh
        } else {
            wh
        }
    }

    fn pdf(&self, wo: &na::Vector3<f32>, wh: &na::Vector3<f32>) -> f32 {
        self.d(wh) * self.g1(wo) * wo.dot(&wh).abs() / abs_cos_theta(wo)
    }
}

pub struct MicrofacetReflection {
    r: Spectrum,
    distribution: Box<dyn MicrofacetDistribution>,
    fresnel: Box<Fresnel>,
}

impl MicrofacetReflection {
    pub fn new(
        r: Spectrum,
        distribution: Box<dyn MicrofacetDistribution>,
        fresnel: Box<Fresnel>,
    ) -> Self {
        Self {
            r,
            distribution,
            fresnel,
        }
    }
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
        _sampled_type: &mut Option<BxDFType>,
    ) -> Spectrum {
        if wo.z == 0. {
            return Spectrum::new(0.);
        }

        let wh = self.distribution.sample_wh(&wo, &u);
        if wo.dot(&wh) < 0. {
            return Spectrum::new(0.);
        }

        *wi = reflect(&wo, &wh);

        if !same_hemisphere(wo, wi) {
            return Spectrum::new(0.);
        }

        *pdf = self.distribution.pdf(wo, &wh) / (4. * wo.dot(&wh));
        self.f(wo, wi)
    }

    fn pdf(&self, wo: &na::Vector3<f32>, wi: &na::Vector3<f32>) -> f32 {
        if !same_hemisphere(wo, wi) {
            return 0.;
        }
        let wh = (wo + wi).normalize();
        self.distribution.pdf(wo, &wh) / (4. * wo.dot(&wh))
    }
}

pub struct MicrofacetTransmission {
    t: Spectrum,
    distribution: Box<dyn MicrofacetDistribution>,
    eta_a: f32,
    eta_b: f32,
    fresnel: FresnelDielectric,
    mode: TransportMode,
}

impl MicrofacetTransmission {
    pub fn new(
        t: Spectrum,
        distribution: Box<dyn MicrofacetDistribution>,
        eta_a: f32,
        eta_b: f32,
        mode: TransportMode,
    ) -> Self {
        Self {
            t,
            distribution,
            eta_a,
            eta_b,
            fresnel: FresnelDielectric::new(eta_a, eta_b),
            mode,
        }
    }
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
        _sampled_type: &mut Option<BxDFType>,
    ) -> Spectrum {
        if wo.z == 0. {
            return Spectrum::new(0.);
        }

        let wh = self.distribution.sample_wh(&wo, &u);
        if wo.dot(&wh) < 0. {
            return Spectrum::new(0.);
        }

        let eta = if cos_theta(wo) > 0. {
            self.eta_a / self.eta_b
        } else {
            self.eta_b / self.eta_a
        };
        if !refract(wo, &wh, eta, wi) {
            return Spectrum::new(0.);
        }
        *pdf = self.pdf(wo, wi);
        self.f(wo, wi)
    }

    fn pdf(&self, wo: &na::Vector3<f32>, wi: &na::Vector3<f32>) -> f32 {
        if !same_hemisphere(wo, wi) {
            return 0.;
        }

        let eta = if cos_theta(wo) > 0. {
            self.eta_a / self.eta_b
        } else {
            self.eta_b / self.eta_a
        };
        let wh = (wo + wi * eta).normalize();

        if wo.dot(&wh) * wi.dot(&wh) > 0. {
            return 0.;
        }

        let sqrt_denom = wo.dot(&wh) + eta * wi.dot(&wh);
        let dwh_dwi = ((eta * eta * wi.dot(&wh)) / (sqrt_denom * sqrt_denom)).abs();

        self.distribution.pdf(wo, &wh) * dwh_dwi
    }
}

pub struct FresnelBlend {
    rd: Spectrum,
    rs: Spectrum,
    distribution: Box<dyn MicrofacetDistribution>,
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
