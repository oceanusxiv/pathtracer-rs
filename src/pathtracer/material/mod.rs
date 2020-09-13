pub mod disney;

use super::{
    bsdf::BSDF,
    bxdf::{
        fresnel::{
            Fresnel, FresnelDielectric, FresnelNoOp, FresnelSpecular, SpecularReflection,
            SpecularTransmission,
        },
        BxDF, LambertianReflection,
    },
    texture::SyncTexture,
    SurfaceMediumInteraction, TransportMode,
};
use crate::common::{self, spectrum::Spectrum};
use ambassador::{delegatable_trait, Delegate};
use common::math::coordinate_system;

#[delegatable_trait]
pub trait MaterialInterface {
    fn compute_scattering_functions(&self, si: &mut SurfaceMediumInteraction, mode: TransportMode);
}

#[derive(Delegate)]
#[delegate(MaterialInterface)]
pub enum Material {
    Matte(MatteMaterial),
    Mirror(MirrorMaterial),
    Glass(GlassMaterial),
}

// FIXME: definitely something wrong with the TBN calculations, normals not correct
pub fn normal_mapping(
    log: &slog::Logger,
    d: &Box<dyn SyncTexture<na::Vector3<f32>>>,
    si: &mut SurfaceMediumInteraction,
) {
    trace!(
        log,
        "tangent space was: {:?} | {:?}, {:?} | {:?}, {:?} | {:?}",
        si.shading.dpdu,
        si.shading.dpdu.norm(),
        si.shading.dpdv,
        si.shading.dpdv.norm(),
        si.shading.n,
        si.shading.n.norm(),
    );
    let tbn = na::Matrix3::from_columns(&[si.shading.dpdu, si.shading.dpdv, si.shading.n]);
    let texture_n = d.evaluate(&si).normalize();
    let ns = (tbn * texture_n).normalize();
    let mut ss = si.shading.dpdu;
    let mut ts = ss.cross(&ns);
    if ts.norm_squared() > 0.0 {
        ts = ts.normalize();
        ss = ts.cross(&ns);
    } else {
        coordinate_system(&ns, &mut ss, &mut ts);
    }

    si.shading.n = ns;
    si.shading.dpdu = ss;
    si.shading.dpdv = ts;
    trace!(
        log,
        "tangent space is now: {:?} | {:?}, {:?} | {:?}, {:?} | {:?}",
        si.shading.dpdu,
        si.shading.dpdu.norm(),
        si.shading.dpdv,
        si.shading.dpdv.norm(),
        si.shading.n,
        si.shading.n.norm(),
    );
}

pub struct MatteMaterial {
    kd: Box<dyn SyncTexture<Spectrum>>,
    normal_map: Option<Box<dyn SyncTexture<na::Vector3<f32>>>>,
    log: slog::Logger,
}

impl MatteMaterial {
    pub fn new(
        log: &slog::Logger,
        kd: Box<dyn SyncTexture<Spectrum>>,
        normal_map: Option<Box<dyn SyncTexture<na::Vector3<f32>>>>,
    ) -> Self {
        let log = log.new(o!());
        Self {
            kd,
            normal_map,
            log,
        }
    }
}

impl MaterialInterface for MatteMaterial {
    fn compute_scattering_functions(
        &self,
        mut si: &mut SurfaceMediumInteraction,
        _mode: TransportMode,
    ) {
        if let Some(normal_map) = self.normal_map.as_ref() {
            normal_mapping(&self.log, normal_map, &mut si);
        }

        let mut bsdf = BSDF::new(&self.log, &si, 1.0);
        let r = self.kd.evaluate(&si);
        bsdf.add(BxDF::Lambertian(LambertianReflection::new(r)));

        si.bsdf = Some(bsdf);
    }
}

pub struct MirrorMaterial {
    log: slog::Logger,
}

impl MirrorMaterial {
    pub fn new(log: &slog::Logger) -> Self {
        let log = log.new(o!());
        Self { log }
    }
}

impl MaterialInterface for MirrorMaterial {
    fn compute_scattering_functions(
        &self,
        si: &mut SurfaceMediumInteraction,
        _mode: TransportMode,
    ) {
        let mut bsdf = BSDF::new(&self.log, &si, 1.0);
        let r = Spectrum::new(1.0);
        bsdf.add(BxDF::SpecularReflection(SpecularReflection::new(
            r,
            Fresnel::NoOp(FresnelNoOp {}),
        )));

        si.bsdf = Some(bsdf);
    }
}

pub struct GlassMaterial {
    kr: Box<dyn SyncTexture<Spectrum>>,
    kt: Box<dyn SyncTexture<Spectrum>>,
    index: Box<dyn SyncTexture<f32>>,
    log: slog::Logger,
}

impl GlassMaterial {
    pub fn new(
        log: &slog::Logger,
        kr: Box<dyn SyncTexture<Spectrum>>,
        kt: Box<dyn SyncTexture<Spectrum>>,
        index: Box<dyn SyncTexture<f32>>,
    ) -> Self {
        let log = log.new(o!());
        Self { kr, kt, index, log }
    }
}

impl MaterialInterface for GlassMaterial {
    fn compute_scattering_functions(&self, si: &mut SurfaceMediumInteraction, mode: TransportMode) {
        let eta = self.index.evaluate(&si);
        let r = self.kr.evaluate(&si);
        let t = self.kt.evaluate(&si);

        let mut bsdf = BSDF::new(&self.log, &si, eta);
        if r.is_black() && t.is_black() {
            return;
        }

        let is_specular = true; // TODO: add roughness factors

        if is_specular {
            bsdf.add(BxDF::FresnelSpecular(FresnelSpecular::new(
                r, t, 1.0, eta, mode,
            )));
        } else {
            if !r.is_black() {
                let fresnel = Fresnel::Dielectric(FresnelDielectric::new(1.0, eta));
                if is_specular {
                    bsdf.add(BxDF::SpecularReflection(SpecularReflection::new(
                        r, fresnel,
                    )));
                } else {
                }
            }

            if !t.is_black() {
                if is_specular {
                    bsdf.add(BxDF::SpecularTransmission(SpecularTransmission::new(
                        t, 1.0, eta, mode,
                    )));
                } else {
                }
            }
        }

        si.bsdf = Some(bsdf);
    }
}
