use super::{
    bsdf::BSDF,
    bxdf::{
        BxDF, Fresnel, FresnelDielectric, FresnelNoOp, FresnelSpecular, LambertianReflection,
        SpecularReflection, SpecularTransmission,
    },
    texture::{ConstantTexture, ImageTexture, NormalMap, SyncTexture, Texture, UVMap},
    SurfaceInteraction, TransportMode,
};
use crate::common::{self, spectrum::Spectrum};
use ambassador::{delegatable_trait, Delegate};
use common::math::coordinate_system;

#[delegatable_trait]
pub trait MaterialInterface {
    fn compute_scattering_functions(&self, si: &mut SurfaceInteraction, mode: TransportMode);
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
    si: &mut SurfaceInteraction,
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
    Kd: Box<dyn SyncTexture<Spectrum>>,
    normal_map: Option<Box<dyn SyncTexture<na::Vector3<f32>>>>,
    log: slog::Logger,
}

impl MatteMaterial {
    pub fn new(
        log: &slog::Logger,
        Kd: Box<dyn SyncTexture<Spectrum>>,
        normal_map: Option<Box<dyn SyncTexture<na::Vector3<f32>>>>,
    ) -> Self {
        let log = log.new(o!());
        Self {
            Kd,
            normal_map,
            log,
        }
    }
}

impl MaterialInterface for MatteMaterial {
    fn compute_scattering_functions(&self, mut si: &mut SurfaceInteraction, _mode: TransportMode) {
        if let Some(normal_map) = self.normal_map.as_ref() {
            normal_mapping(&self.log, normal_map, &mut si);
        }

        let mut bsdf = BSDF::new(&self.log, &si, 1.0);
        let r = self.Kd.evaluate(&si);
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
    fn compute_scattering_functions(&self, si: &mut SurfaceInteraction, _mode: TransportMode) {
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
    Kr: Box<dyn SyncTexture<Spectrum>>,
    Kt: Box<dyn SyncTexture<Spectrum>>,
    index: Box<dyn SyncTexture<f32>>,
    log: slog::Logger,
}

impl GlassMaterial {
    pub fn new(
        log: &slog::Logger,
        Kr: Box<dyn SyncTexture<Spectrum>>,
        Kt: Box<dyn SyncTexture<Spectrum>>,
        index: Box<dyn SyncTexture<f32>>,
    ) -> Self {
        let log = log.new(o!());
        Self { Kr, Kt, index, log }
    }
}

impl MaterialInterface for GlassMaterial {
    fn compute_scattering_functions(&self, si: &mut SurfaceInteraction, mode: TransportMode) {
        let eta = self.index.evaluate(&si);
        let R = self.Kr.evaluate(&si);
        let T = self.Kt.evaluate(&si);

        let mut bsdf = BSDF::new(&self.log, &si, eta);
        if R.is_black() && T.is_black() {
            return;
        }

        let is_specular = true;

        // FIXME: enable fresnel specular when monte carlo is ready
        if is_specular && false {
            bsdf.add(BxDF::FresnelSpecular(FresnelSpecular::new(
                R, T, 1.0, eta, mode,
            )));
        } else {
            if !R.is_black() {
                let fresnel = Fresnel::Dielectric(FresnelDielectric::new(1.0, eta));
                if is_specular {
                    bsdf.add(BxDF::SpecularReflection(SpecularReflection::new(
                        R, fresnel,
                    )));
                } else {
                }
            }

            if !T.is_black() {
                if is_specular {
                    bsdf.add(BxDF::SpecularTransmission(SpecularTransmission::new(
                        T, 1.0, eta, mode,
                    )));
                } else {
                }
            }
        }

        si.bsdf = Some(bsdf);
    }
}
