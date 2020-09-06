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

impl Material {
    pub fn from_gltf(log: &slog::Logger, gltf_material: &common::Material) -> Self {
        let pbr = &gltf_material.pbr_metallic_roughness;
        let color_factor = Spectrum::from_slice(&pbr.base_color_factor, true);
        let color_texture;
        let normal_map;
        if let Some(color_info) = pbr.color_texture.as_ref() {
            color_texture = Box::new(ImageTexture::<Spectrum>::new(
                log,
                &color_info.image,
                color_factor,
                color_info.sampler_info.wrap_mode,
                UVMap::new(
                    color_info.image.width() as f32,
                    color_info.image.height() as f32,
                    0.0,
                    0.0,
                ),
                true,
            )) as Box<dyn SyncTexture<Spectrum>>;
        } else {
            color_texture = Box::new(ConstantTexture::<Spectrum>::new(color_factor))
                as Box<dyn SyncTexture<Spectrum>>;
        }

        if let Some(normal_map_info) = pbr.normal_map.as_ref() {
            normal_map = Some(Box::new(NormalMap::new(
                log,
                &normal_map_info.texture.image,
                na::Vector2::new(normal_map_info.scale, normal_map_info.scale),
                normal_map_info.texture.sampler_info.wrap_mode,
                UVMap::new(
                    normal_map_info.texture.image.width() as f32,
                    normal_map_info.texture.image.height() as f32,
                    0.0,
                    0.0,
                ),
            )) as Box<dyn SyncTexture<na::Vector3<f32>>>);
        } else {
            normal_map = None;
        }

        if gltf_material.transmission_factor > 0.0 {
            let index = Box::new(ConstantTexture::<f32>::new(gltf_material.ior))
                as Box<dyn SyncTexture<f32>>;
            let reflect_color = Box::new(ConstantTexture::<Spectrum>::new(Spectrum::new(1.0)))
                as Box<dyn SyncTexture<Spectrum>>;
            let transmit_color = Box::new(ConstantTexture::<Spectrum>::new(Spectrum::new(1.0)))
                as Box<dyn SyncTexture<Spectrum>>;
            return Material::Glass(GlassMaterial::new(
                log,
                reflect_color,
                transmit_color,
                index,
            ));
        }

        if pbr.metallic_factor == 1.0 && pbr.roughness_factor == 0.0 {
            return Material::Mirror(MirrorMaterial::new(log));
        }

        if pbr.metallic_factor == 0.0 && pbr.roughness_factor == 0.0 {
            return Material::Matte(MatteMaterial::new(log, color_texture, normal_map));
        }

        Material::Matte(MatteMaterial::new(log, color_texture, normal_map))
    }
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
