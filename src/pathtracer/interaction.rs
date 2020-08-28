use super::{material::BSDF, primitive::Primitive, shape::Shape};
use std::sync::Arc;

pub struct SurfaceInteractionShading {
    pub n: na::Vector3<f32>,
}

pub struct SurfaceInteraction<'a> {
    pub p: na::Point3<f32>,
    pub p_error: na::Vector3<f32>,
    pub wo: na::Vector3<f32>,
    pub n: na::Vector3<f32>,
    pub shading: SurfaceInteractionShading,
    pub shape: Option<&'a dyn Shape>,
    pub primitive: Option<&'a dyn Primitive>,
    pub bsdf: Option<&'a dyn BSDF>,
}

impl<'a> SurfaceInteraction<'a> {
    pub fn empty() -> Self {
        SurfaceInteraction {
            p: na::Point3::new(0.0, 0.0, 0.0),
            p_error: glm::zero(),
            wo: glm::zero(),
            n: glm::zero(),
            shading: SurfaceInteractionShading { n: glm::zero() },
            shape: None,
            primitive: None,
            bsdf: None,
        }
    }

    // pub fn new(
    //     p: &na::Point3<f32>,
    //     p_error: &na::Vector3<f32>,
    //     uv: &na::Point2<f32>,
    //     wo: &na::Vector3<f32>,
    //     dpdu: &na::Vector3<f32>,
    //     dpdv: &na::Vector3<f32>,
    //     dndu: &na::Vector3<f32>,
    //     dndv: &na::Vector3<f32>,
    //     shape: Box<dyn Shape>,
    // ) -> Self {
    // }
}
