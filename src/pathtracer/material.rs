use super::{
    bsdf::BSDF,
    bxdf::{BxDF, Fresnel, FresnelNoOp, LambertianReflection, SpecularReflection},
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
        if gltf_material.pbr_metallic_roughness.metallic_factor == 1.0
            && gltf_material.pbr_metallic_roughness.roughness_factor == 0.0
        {
            return Material::Mirror(MirrorMaterial {});
        }

        if gltf_material.pbr_metallic_roughness.metallic_factor == 0.0
            && gltf_material.pbr_metallic_roughness.roughness_factor == 0.0
        {
            return Material::Matte(MatteMaterial {});
        }

        Material::Matte(MatteMaterial {})
    }
}

pub struct MatteMaterial {}

impl MaterialInterface for MatteMaterial {
    fn compute_scattering_functions(&self, si: &mut SurfaceInteraction, mode: TransportMode) {
        let mut bsdf = BSDF::new(&si, 1.0);
        let r = Spectrum::new(1.0);
        bsdf.add(BxDF::Lambertian(LambertianReflection::new(r)));

        si.bsdf = Some(bsdf);
    }
}

pub struct MirrorMaterial {}

impl MaterialInterface for MirrorMaterial {
    fn compute_scattering_functions(&self, si: &mut SurfaceInteraction, mode: TransportMode) {
        let mut bsdf = BSDF::new(&si, 1.0);
        let r = Spectrum::new(1.0);
        bsdf.add(BxDF::SpecularReflection(SpecularReflection::new(
            r,
            Fresnel::NoOp(FresnelNoOp {}),
        )));

        si.bsdf = Some(bsdf);
    }
}
