use crate::{
    common::{math::lerp, spectrum::Spectrum},
    pathtracer::bsdf::BSDF,
    pathtracer::bxdf::{
        abs_cos_theta,
        fresnel::{fr_dielectric, Fresnel, FresnelInterface},
        microfacet::{
            MicrofacetDistribution, MicrofacetReflection, MicrofacetTransmission,
            TrowbridgeReitzDistribution,
        },
        BxDF, BxDFInterface, BxDFType,
    },
    pathtracer::interaction::SurfaceMediumInteraction,
    pathtracer::texture::SyncTexture,
    pathtracer::TransportMode,
};

use super::{normal_mapping, MaterialInterface};

pub struct DisneyMaterial {
    color: Box<dyn SyncTexture<Spectrum>>,
    metallic: Box<dyn SyncTexture<f32>>,
    eta: Box<dyn SyncTexture<f32>>,
    roughness: Box<dyn SyncTexture<f32>>,
    thin: bool,
    normal_map: Option<Box<dyn SyncTexture<na::Vector3<f32>>>>,
    log: slog::Logger,
}

// TODO: specular tint
// TODO: anisotropic
// TODO: sheen
// TODO: sheen tint
// TODO: clear coat
// TODO: clear coat gloss
// TODO: specular trans
// TODO: scatter distance
// TODO: thin
// TODO: flatness
// TODO: diffuse trans
impl DisneyMaterial {
    pub fn new(
        log: &slog::Logger,
        color: Box<dyn SyncTexture<Spectrum>>,
        metallic: Box<dyn SyncTexture<f32>>,
        eta: Box<dyn SyncTexture<f32>>,
        roughness: Box<dyn SyncTexture<f32>>,
        normal_map: Option<Box<dyn SyncTexture<na::Vector3<f32>>>>,
    ) -> Self {
        let log = log.new(o!());
        Self {
            color,
            metallic,
            eta,
            roughness,
            thin: false,
            normal_map,
            log,
        }
    }
}

fn sqr(x: f32) -> f32 {
    x * x
}

fn schlick_weight(cos_theta: f32) -> f32 {
    let m = (1.0 - cos_theta).clamp(0.0, 1.0);
    (m * m) * (m * m) * m
}

fn fr_schlick(r0: f32, cos_theta: f32) -> f32 {
    lerp(r0, 1.0, schlick_weight(cos_theta))
}

fn fr_schlick_spectrum(r0: &Spectrum, cos_theta: f32) -> Spectrum {
    lerp(*r0, Spectrum::new(1.), schlick_weight(cos_theta))
}

fn schlick_r0_from_eta(eta: f32) -> f32 {
    sqr(eta - 1.0) / sqr(eta + 1.0)
}

pub struct DisneyDiffuse {
    r: Spectrum,
}

impl DisneyDiffuse {
    pub fn new(r: Spectrum) -> Self {
        Self { r }
    }
}

impl BxDFInterface for DisneyDiffuse {
    fn f(&self, wo: &na::Vector3<f32>, wi: &na::Vector3<f32>) -> Spectrum {
        let fo = schlick_weight(abs_cos_theta(&wo));
        let fi = schlick_weight(abs_cos_theta(&wi));

        // Diffuse fresnel - go from 1 at normal incidence to .5 at grazing.
        // Burley 2015, eq (4).
        self.r * std::f32::consts::FRAC_1_PI * (1. - fo / 2.) * (1. - fi / 2.)
    }

    fn rho(
        &self,
        _wo: &na::Vector3<f32>,
        _n_samples: usize,
        _samples: &[na::Point2<f32>],
    ) -> Spectrum {
        self.r
    }

    fn rho_no_wo(
        &self,
        _n_samples: usize,
        _samples_1: &[na::Point2<f32>],
        _samples_2: &[na::Point2<f32>],
    ) -> Spectrum {
        self.r
    }

    fn get_type(&self) -> BxDFType {
        BxDFType::BSDF_REFLECTION | BxDFType::BSDF_DIFFUSE
    }
}

pub struct DisneyFresnel {
    r0: Spectrum,
    metallic: f32,
    eta: f32,
}

impl DisneyFresnel {
    pub fn new(r0: Spectrum, metallic: f32, eta: f32) -> Self {
        Self { r0, metallic, eta }
    }
}

impl FresnelInterface for DisneyFresnel {
    fn evaluate(&self, cos_i: f32) -> Spectrum {
        lerp(
            Spectrum::new(fr_dielectric(cos_i, 1., self.eta)),
            fr_schlick_spectrum(&self.r0, cos_i),
            self.metallic,
        )
    }
}

struct DisneyMicrofacetDistribution {
    distribution: TrowbridgeReitzDistribution,
}

impl DisneyMicrofacetDistribution {
    pub fn new(alpha_x: f32, alpha_y: f32) -> Self {
        Self {
            distribution: TrowbridgeReitzDistribution::new(alpha_x, alpha_y),
        }
    }
}

impl MicrofacetDistribution for DisneyMicrofacetDistribution {
    fn d(&self, wh: &na::Vector3<f32>) -> f32 {
        self.distribution.d(&wh)
    }

    fn lambda(&self, w: &na::Vector3<f32>) -> f32 {
        self.distribution.lambda(&w)
    }

    fn g(&self, wo: &na::Vector3<f32>, wi: &na::Vector3<f32>) -> f32 {
        self.g1(&wo) * self.g1(&wi)
    }

    fn sample_wh(&self, wo: &na::Vector3<f32>, u: &na::Point2<f32>) -> na::Vector3<f32> {
        self.distribution.sample_wh(&wo, &u)
    }

    fn pdf(&self, wo: &na::Vector3<f32>, wh: &na::Vector3<f32>) -> f32 {
        self.distribution.pdf(&wo, &wh)
    }
}

impl MaterialInterface for DisneyMaterial {
    fn compute_scattering_functions(
        &self,
        mut si: &mut SurfaceMediumInteraction,
        mode: TransportMode,
    ) {
        if let Some(normal_map) = self.normal_map.as_ref() {
            normal_mapping(&self.log, normal_map, &mut si);
        }

        let mut bsdf = BSDF::new(&self.log, &si, 1.0);

        let c = self.color.evaluate(&si);
        let metallic_weight = self.metallic.evaluate(&si);
        let e = self.eta.evaluate(&si);
        let strans = 0.0;
        let diffuse_weight = (1.0 - metallic_weight) * (1.0 - strans);
        let dt = 0.0;
        let rough = self.roughness.evaluate(&si);
        let lum = c.y();
        let c_tint = if lum > 0.0 {
            c / lum
        } else {
            Spectrum::new(1.0)
        };

        let sheen_weight = 0.0;
        let c_sheen = if sheen_weight > 0.0 {
            let stint = 0.0;
            lerp(Spectrum::new(1.), c_tint, stint)
        } else {
            Spectrum::new(0.0)
        };

        if diffuse_weight > 0.0 {
            // TODO: thin
            if self.thin {
                panic!("thin not supported!");
            } else {
                // TODO: subsurface scattering
                let sd = Spectrum::new(0.0);
                if sd.is_black() {
                    bsdf.add(BxDF::DisneyDiffuse(DisneyDiffuse::new(diffuse_weight * c)));
                } else {
                    panic!("subsurface scattering not supported!");
                }
            }

            //TODO: retro-reflection
            //TODO: sheen
            if sheen_weight > 0.0 {
                panic!("sheen not supported!");
            }
        }

        // TODO: anisotrophy
        let aspect = 1.0;
        let ax = 0.001f32.max(sqr(rough) / aspect);
        let ay = 0.001f32.max(sqr(rough) * aspect);
        // TODO: think about using Arc instead of Box to save memory allocations, or even ref

        // TODO: specular tint
        let spec_tint = 0.0;
        let c_spec_0 = lerp(
            schlick_r0_from_eta(e) * lerp(Spectrum::new(1.), c_tint, spec_tint),
            c,
            metallic_weight,
        );
        bsdf.add(BxDF::MicrofacetReflection(MicrofacetReflection::new(
            Spectrum::new(1.),
            Box::new(DisneyMicrofacetDistribution::new(ax, ay)),
            Box::new(Fresnel::Disney(DisneyFresnel::new(
                c_spec_0,
                metallic_weight,
                e,
            ))),
        )));

        // clear coat

        if strans > 0.0 {
            let t = strans * c.sqrt();
            if self.thin {
                panic!("thin not supported!");
            } else {
                bsdf.add(BxDF::MicrofacetTransmission(MicrofacetTransmission::new(
                    t,
                    Box::new(DisneyMicrofacetDistribution::new(ax, ay)),
                    1.0,
                    e,
                    mode,
                )));
            }
        }
        if self.thin {
            panic!("thin not supported!");
        }

        si.bsdf = Some(bsdf);
    }
}
