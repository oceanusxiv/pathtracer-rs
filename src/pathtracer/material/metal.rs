use super::MaterialInterface;
use crate::common::spectrum::Spectrum;
use crate::pathtracer::{
    bsdf::BSDF,
    bxdf::{
        fresnel::{Fresnel, FresnelConductor},
        microfacet::{MicrofacetReflection, TrowbridgeReitzDistribution},
        BxDF,
    },
    texture::SyncTexture,
    SurfaceMediumInteraction, TransportMode,
};
pub struct MetalMaterial {
    eta: Box<dyn SyncTexture<Spectrum>>,
    k: Box<dyn SyncTexture<Spectrum>>,
    r: Box<dyn SyncTexture<Spectrum>>,
    roughness: Option<Box<dyn SyncTexture<f32>>>,
    u_roughness: Option<Box<dyn SyncTexture<f32>>>,
    v_roughness: Option<Box<dyn SyncTexture<f32>>>,
    remap_roughness: bool,
    log: slog::Logger,
}

impl MetalMaterial {
    pub fn new(
        log: &slog::Logger,
        eta: Box<dyn SyncTexture<Spectrum>>,
        k: Box<dyn SyncTexture<Spectrum>>,
        r: Box<dyn SyncTexture<Spectrum>>,
        roughness: Option<Box<dyn SyncTexture<f32>>>,
        u_roughness: Option<Box<dyn SyncTexture<f32>>>,
        v_roughness: Option<Box<dyn SyncTexture<f32>>>,
        remap_roughness: bool,
    ) -> Self {
        let log = log.new(o!());
        Self {
            eta,
            k,
            r,
            roughness,
            u_roughness,
            v_roughness,
            remap_roughness,
            log,
        }
    }
}

impl MaterialInterface for MetalMaterial {
    fn compute_scattering_functions(
        &self,
        si: &mut SurfaceMediumInteraction,
        _mode: TransportMode,
    ) {
        let mut bsdf = BSDF::new(&self.log, &si, 1.0);

        let mut u_rough = if let Some(u_roughness) = self.u_roughness.as_ref() {
            u_roughness.evaluate(&si)
        } else {
            if let Some(roughness) = self.roughness.as_ref() {
                roughness.evaluate(&si)
            } else {
                panic!("neither u roughness nor roughness specified");
            }
        };

        let mut v_rough = if let Some(v_roughness) = self.v_roughness.as_ref() {
            v_roughness.evaluate(&si)
        } else {
            if let Some(roughness) = self.roughness.as_ref() {
                roughness.evaluate(&si)
            } else {
                panic!("neither v roughness nor roughness specified");
            }
        };

        if self.remap_roughness {
            u_rough = TrowbridgeReitzDistribution::roughness_to_alpha(u_rough);
            v_rough = TrowbridgeReitzDistribution::roughness_to_alpha(v_rough);
        }

        bsdf.add(BxDF::MicrofacetReflection(MicrofacetReflection::new(
            self.r.evaluate(&si),
            Box::new(TrowbridgeReitzDistribution::new(u_rough, v_rough)),
            Box::new(Fresnel::Conductor(FresnelConductor::new(
                Spectrum::new(1.),
                self.eta.evaluate(&si),
                self.k.evaluate(&si),
            ))),
        )));

        si.bsdf = Some(bsdf);
    }
}
