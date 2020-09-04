use super::{
    bsdf::BSDF,
    bxdf::{BxDF, Fresnel, FresnelNoOp, LambertianReflection, SpecularReflection},
    texture::{ConstantTexture, ImageTexture, NormalMap, SyncTexture, Texture, UVMap},
    SurfaceInteraction, TransportMode,
};
use crate::common::{self, spectrum::Spectrum};
use ambassador::{delegatable_trait, Delegate};

#[delegatable_trait]
pub trait MaterialInterface {
    fn compute_scattering_functions(&self, si: &mut SurfaceInteraction, mode: TransportMode);
}

#[derive(Delegate)]
#[delegate(MaterialInterface)]
pub enum Material {
    Matte(MatteMaterial),
    Mirror(MirrorMaterial),
}

impl Material {
    pub fn from_gltf(gltf_material: &common::Material) -> Self {
        let pbr = &gltf_material.pbr_metallic_roughness;
        let color_factor = Spectrum::from_slice(&pbr.base_color_factor, true);
        let color_texture;
        let normal_map;
        if let Some(color_info) = pbr.color_texture.as_ref() {
            color_texture = Box::new(ImageTexture::<Spectrum>::new(
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
                &normal_map_info.texture.image,
                normal_map_info.scale,
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

        if pbr.metallic_factor == 1.0 && pbr.roughness_factor == 0.0 {
            return Material::Mirror(MirrorMaterial {});
        }

        if pbr.metallic_factor == 0.0 && pbr.roughness_factor == 0.0 {
            return Material::Matte(MatteMaterial::new(color_texture, normal_map));
        }

        Material::Matte(MatteMaterial::new(color_texture, normal_map))
    }
}

pub fn normal_mapping(d: &Box<dyn SyncTexture<na::Vector3<f32>>>, si: &mut SurfaceInteraction) {
    trace!("normal was: {:?}", si.shading.n);
    let texture_n = d.evaluate(&si).normalize();
    let n_diff = texture_n - na::Vector3::new(0.0, 0.0, 1.0);
    si.shading.n = (si.shading.n + n_diff).normalize();
    si.shading.dpdu = (si.shading.dpdu + n_diff).normalize();
    si.shading.dpdv = (si.shading.dpdv + n_diff).normalize();
    trace!("normal is now: {:?}", si.shading.n);
}

pub struct MatteMaterial {
    Kd: Box<dyn SyncTexture<Spectrum>>,
    normal_map: Option<Box<dyn SyncTexture<na::Vector3<f32>>>>,
}

impl MatteMaterial {
    pub fn new(
        Kd: Box<dyn SyncTexture<Spectrum>>,
        normal_map: Option<Box<dyn SyncTexture<na::Vector3<f32>>>>,
    ) -> Self {
        Self { Kd, normal_map }
    }
}

impl MaterialInterface for MatteMaterial {
    fn compute_scattering_functions(&self, mut si: &mut SurfaceInteraction, _mode: TransportMode) {
        if let Some(normal_map) = self.normal_map.as_ref() {
            normal_mapping(normal_map, &mut si);
        }

        let mut bsdf = BSDF::new(&si, 1.0);
        let r = self.Kd.evaluate(&si);
        bsdf.add(BxDF::Lambertian(LambertianReflection::new(r)));

        si.bsdf = Some(bsdf);
    }
}

pub struct MirrorMaterial {}

impl MaterialInterface for MirrorMaterial {
    fn compute_scattering_functions(&self, si: &mut SurfaceInteraction, _mode: TransportMode) {
        let mut bsdf = BSDF::new(&si, 1.0);
        let r = Spectrum::new(1.0);
        bsdf.add(BxDF::SpecularReflection(SpecularReflection::new(
            r,
            Fresnel::NoOp(FresnelNoOp {}),
        )));

        si.bsdf = Some(bsdf);
    }
}
