use super::{
    bsdf::BSDF,
    bxdf::{BxDF, Fresnel, FresnelNoOp, LambertianReflection, SpecularReflection},
    texture::{ConstantTexture, ImageTexture, SyncTexture, Texture, UVMap},
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
        let color_factor = Spectrum::from_slice(&pbr.base_color_factor);
        let color_texture;
        if let Some(color_info) = pbr.color_texture.as_ref() {
            color_texture = Box::new(ImageTexture::<Spectrum>::new(
                &color_info.image,
                color_factor,
                color_info.sampler_info.wrap_mode,
                UVMap::new(color_info.image.width() as f32, color_info.image.height() as f32, 0.0, 0.0),
            )) as Box<dyn SyncTexture<Spectrum>>;
        } else {
            color_texture = Box::new(ConstantTexture::<Spectrum>::new(color_factor))
                as Box<dyn SyncTexture<Spectrum>>;
        }
        if pbr.metallic_factor == 1.0 && pbr.roughness_factor == 0.0 {
            return Material::Mirror(MirrorMaterial {});
        }

        if pbr.metallic_factor == 0.0 && pbr.roughness_factor == 0.0 {
            return Material::Matte(MatteMaterial::new(color_texture));
        }

        Material::Matte(MatteMaterial::new(color_texture))
    }
}

pub struct MatteMaterial {
    Kd: Box<dyn SyncTexture<Spectrum>>,
}

impl MatteMaterial {
    pub fn new(Kd: Box<dyn SyncTexture<Spectrum>>) -> Self {
        Self { Kd }
    }
}

impl MaterialInterface for MatteMaterial {
    fn compute_scattering_functions(&self, si: &mut SurfaceInteraction, _mode: TransportMode) {
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
