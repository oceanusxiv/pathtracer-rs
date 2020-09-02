use super::{bsdf::BSDF, primitive::Primitive, shape::Shape, TransportMode};
use crate::common::{
    math::{face_forward, offset_ray_origin},
    ray::Ray,
    spectrum::Spectrum,
};
use std::{cell::RefCell, sync::Arc};

#[derive(Clone, Copy, Debug)]
pub struct Interaction {
    pub p: na::Point3<f32>,
    pub time: f32,
    pub p_error: na::Vector3<f32>,
    pub wo: na::Vector3<f32>,
    pub n: na::Vector3<f32>,
}

impl Default for Interaction {
    fn default() -> Self {
        Interaction {
            p: na::Point3::origin(),
            time: 0.0,
            p_error: glm::zero(),
            wo: glm::zero(),
            n: glm::zero(),
        }
    }
}

const SHADOW_EPSILON: f32 = 0.0001;

impl Interaction {
    pub fn spawn_ray(&self, d: &na::Vector3<f32>) -> Ray {
        let o = offset_ray_origin(&self.p, &self.p_error, &self.n, d);
        Ray {
            o,
            d: *d,
            t_max: RefCell::new(f32::INFINITY),
        }
    }
    pub fn spawn_ray_to(&self, p2: &na::Point3<f32>) -> Ray {
        let origin = offset_ray_origin(&self.p, &self.p_error, &self.n, &(p2 - self.p));
        let d = p2 - origin;
        Ray {
            o: origin,
            d: d,
            t_max: RefCell::new(1.0 - SHADOW_EPSILON),
        }
    }

    pub fn spawn_ray_to_it(&self, it2: &Interaction) -> Ray {
        let origin = offset_ray_origin(&self.p, &self.p_error, &self.n, &(it2.p - self.p));
        let target = offset_ray_origin(&it2.p, &it2.p_error, &it2.n, &(origin - it2.p));
        let d = target - origin;
        return Ray {
            o: origin,
            d,
            t_max: RefCell::new(1.0 - SHADOW_EPSILON),
        };
    }
}

pub struct SurfaceInteractionShading {
    pub n: na::Vector3<f32>,
    pub dpdu: na::Vector3<f32>,
    pub dpdv: na::Vector3<f32>,
    pub dndu: na::Vector3<f32>,
    pub dndv: na::Vector3<f32>,
}

impl Default for SurfaceInteractionShading {
    fn default() -> Self {
        SurfaceInteractionShading {
            n: glm::zero(),
            dpdu: glm::zero(),
            dpdv: glm::zero(),
            dndu: glm::zero(),
            dndv: glm::zero(),
        }
    }
}

pub struct SurfaceInteraction<'a> {
    pub general: Interaction,
    pub uv: na::Point2<f32>,
    pub dpdu: na::Vector3<f32>,
    pub dpdv: na::Vector3<f32>,
    pub dndu: na::Vector3<f32>,
    pub dndv: na::Vector3<f32>,
    pub shading: SurfaceInteractionShading,
    pub shape: Option<&'a dyn Shape>,
    pub primitive: Option<&'a dyn Primitive>,
    pub bsdf: Option<BSDF>,

    pub dpdx: na::Vector3<f32>,
    pub dpdy: na::Vector3<f32>,
    pub dudx: f32,
    pub dvdx: f32,
    pub dudy: f32,
    pub dvdy: f32,
}

impl<'a> Default for SurfaceInteraction<'a> {
    fn default() -> SurfaceInteraction<'a> {
        SurfaceInteraction {
            general: Interaction {
                p: na::Point3::origin(),
                time: 0.0,
                p_error: glm::zero(),
                wo: glm::zero(),
                n: glm::zero(),
            },
            uv: na::Point2::new(0.0, 0.0),
            dpdu: glm::zero(),
            dpdv: glm::zero(),
            dndu: glm::zero(),
            dndv: glm::zero(),
            shading: Default::default(),
            shape: None,
            primitive: None,
            bsdf: None,
            dpdx: glm::zero(),
            dpdy: glm::zero(),
            dudx: 0.0,
            dvdx: 0.0,
            dudy: 0.0,
            dvdy: 0.0,
        }
    }
}

impl<'a> SurfaceInteraction<'a> {
    pub fn new(
        p: &na::Point3<f32>,
        p_error: &na::Vector3<f32>,
        uv: &na::Point2<f32>,
        wo: &na::Vector3<f32>,
        dpdu: &na::Vector3<f32>,
        dpdv: &na::Vector3<f32>,
        dndu: &na::Vector3<f32>,
        dndv: &na::Vector3<f32>,
        time: f32,
        shape: &'a dyn Shape,
    ) -> Self {
        let n = dpdu.cross(dpdv).normalize();
        let shading = SurfaceInteractionShading {
            n,
            dpdu: *dpdu,
            dpdv: *dpdv,
            dndu: *dndu,
            dndv: *dndv,
        };
        SurfaceInteraction {
            general: Interaction {
                p: *p,
                time,
                p_error: *p_error,
                wo: *wo,
                n,
            },
            uv: *uv,
            dpdu: *dpdu,
            dpdv: *dpdv,
            dndu: *dndu,
            dndv: *dndv,
            shading,
            shape: Some(shape),
            primitive: None,
            bsdf: None,
            ..Default::default()
        }
    }

    pub fn set_shading_geometry(
        &mut self,
        dpdus: &na::Vector3<f32>,
        dpdvs: &na::Vector3<f32>,
        dndus: &na::Vector3<f32>,
        dndvs: &na::Vector3<f32>,
        orientation_is_authoritative: bool,
    ) {
        // Compute _shading.n_ for _SurfaceInteraction_
        self.shading.n = dpdus.cross(dpdvs).normalize();
        if orientation_is_authoritative {
            self.general.n = face_forward(&self.general.n, &self.shading.n);
        } else {
            self.shading.n = face_forward(&self.shading.n, &self.general.n);
        }

        // Initialize _shading_ partial derivative values
        self.shading.dpdu = *dpdus;
        self.shading.dpdv = *dpdvs;
        self.shading.dndu = *dndus;
        self.shading.dndv = *dndvs;
    }

    pub fn compute_scattering_functions(&mut self, r: &Ray, mode: TransportMode) {
        self.primitive
            .unwrap()
            .compute_scattering_functions(self, mode);
    }

    pub fn le(&self, w: &na::Vector3<f32>) -> Spectrum {
        Spectrum::new(0.0)
    }
}
