use super::MaterialInterface;
use crate::common::spectrum::Spectrum;
use crate::pathtracer::{
    bsdf::BSDF,
    bxdf::{
        fresnel::Fresnel,
        microfacet::{FresnelBlend, TrowbridgeReitzDistribution},
        BxDF,
    },
    texture::SyncTexture,
    SurfaceMediumInteraction, TransportMode,
};
pub struct SubstrateMaterial {
    kd: Box<dyn SyncTexture<Spectrum>>,
    ks: Box<dyn SyncTexture<Spectrum>>,
    nu: Box<dyn SyncTexture<f32>>,
    nv: Box<dyn SyncTexture<f32>>,
    remap_roughness: bool,
    log: slog::Logger,
}

impl SubstrateMaterial {
    pub fn new(
        log: &slog::Logger,
        kd: Box<dyn SyncTexture<Spectrum>>,
        ks: Box<dyn SyncTexture<Spectrum>>,
        nu: Box<dyn SyncTexture<f32>>,
        nv: Box<dyn SyncTexture<f32>>,
        remap_roughness: bool,
    ) -> Self {
        let log = log.new(o!());
        Self {
            kd,
            ks,
            nu,
            nv,
            remap_roughness,
            log,
        }
    }
}

impl MaterialInterface for SubstrateMaterial {
    fn compute_scattering_functions(
        &self,
        mut si: &mut SurfaceMediumInteraction,
        _mode: TransportMode,
    ) {
        let mut bsdf = BSDF::new(&self.log, &si, 1.0);

        let d = self.kd.evaluate(&si);
        let s = self.ks.evaluate(&si);
        let mut rough_u = self.nu.evaluate(&si);
        let mut rough_v = self.nv.evaluate(&si);

        if !d.is_black() || s.is_black() {
            if self.remap_roughness {
                rough_u = TrowbridgeReitzDistribution::roughness_to_alpha(rough_u);
                rough_v = TrowbridgeReitzDistribution::roughness_to_alpha(rough_v);
            }

            bsdf.add(BxDF::FresnelBlend(FresnelBlend::new(
                d,
                s,
                Box::new(TrowbridgeReitzDistribution::new(rough_u, rough_v)),
            )));
        }
        si.bsdf = Some(bsdf);
    }
}
