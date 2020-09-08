use std::sync::Arc;

use super::{interaction::Interaction, shape::SyncShape, texture::MIPMap, RenderScene};
use crate::common::{
    bounds::Bounds3,
    ray::{Ray, RayDifferential},
    spectrum::Spectrum,
};

bitflags! {
    pub struct LightFlags: u32 {
        const DELTA_POSITION = 1;
        const DELTA_DIRECTION = 2;
        const AREA = 4;
        const INFINITE = 8;
    }
}

pub fn is_delta_light(flags: &LightFlags) -> bool {
    flags.contains(LightFlags::DELTA_DIRECTION) || flags.contains(LightFlags::DELTA_POSITION)
}

pub struct VisibilityTester {
    p0: Interaction,
    p1: Interaction,
}

impl<'a> VisibilityTester {
    pub fn unoccluded(&self, scene: &RenderScene) -> bool {
        !scene.intersect_p(&self.p0.spawn_ray_to_it(&self.p1))
    }
}

pub trait Light {
    fn le(&self, _r: &RayDifferential) -> Spectrum {
        Spectrum::new(0.0)
    }

    fn sample_li(
        &self,
        reference: &Interaction,
        u: &na::Point2<f32>,
        wi: &mut na::Vector3<f32>,
        pdf: &mut f32,
        vis: &mut Option<VisibilityTester>,
    ) -> Spectrum;

    fn power(&self) -> Spectrum;

    fn pdf_li(&self, reference: &Interaction, wi: &na::Vector3<f32>) -> f32;

    fn sample_le(
        &self,
        u1: &na::Point2<f32>,
        u2: &na::Point2<f32>,
        r: &mut Ray,
        n_light: &na::Vector3<f32>,
        pdf_pos: &mut f32,
        pdf_dir: &mut f32,
    );

    fn pdf_le(&self, r: &Ray, n_light: &na::Vector3<f32>, pdf_pos: &mut f32, pdf_dir: &mut f32);

    fn preprocess(&mut self, world_bound: &Bounds3) {}
}

pub trait SyncLight: Light + Send + Sync {}
impl<T> SyncLight for T where T: Light + Send + Sync {}

pub struct PointLight {
    flags: LightFlags,
    num_samples: u32,
    light_to_world: na::Projective3<f32>,
    world_to_light: na::Projective3<f32>,
    p_light: na::Point3<f32>,
    I: Spectrum,
}

impl PointLight {
    pub fn new(light_to_world: &na::Projective3<f32>, I: Spectrum) -> Self {
        Self {
            flags: LightFlags::DELTA_POSITION,
            num_samples: 1,
            light_to_world: *light_to_world,
            world_to_light: light_to_world.inverse(),
            p_light: light_to_world * na::Point3::origin(),
            I,
        }
    }
}

impl Light for PointLight {
    fn sample_li(
        &self,
        reference: &Interaction,
        u: &na::Point2<f32>,
        wi: &mut na::Vector3<f32>,
        pdf: &mut f32,
        vis: &mut Option<VisibilityTester>,
    ) -> Spectrum {
        *wi = (self.p_light - reference.p).normalize();
        *pdf = 1.0;
        *vis = Some(VisibilityTester {
            p0: *reference,
            p1: Interaction {
                p: self.p_light,
                time: reference.time,
                ..Default::default()
            },
        });

        self.I / (self.p_light - reference.p).norm_squared()
    }

    fn power(&self) -> Spectrum {
        4.0 * std::f32::consts::PI * self.I
    }

    fn pdf_li(&self, reference: &Interaction, wi: &na::Vector3<f32>) -> f32 {
        0.0
    }

    fn sample_le(
        &self,
        u1: &na::Point2<f32>,
        u2: &na::Point2<f32>,
        r: &mut Ray,
        n_light: &na::Vector3<f32>,
        pdf_pos: &mut f32,
        pdf_dir: &mut f32,
    ) {
        todo!()
    }

    fn pdf_le(&self, r: &Ray, n_light: &na::Vector3<f32>, pdf_pos: &mut f32, pdf_dir: &mut f32) {
        todo!()
    }
}

pub struct DirectionalLight {
    flags: LightFlags,
    light_to_world: na::Projective3<f32>,
    world_to_light: na::Projective3<f32>,
    L: Spectrum,
    w_light: na::Vector3<f32>,
    world_center: na::Point3<f32>,
    world_radius: f32,
}

impl DirectionalLight {
    pub fn new(
        light_to_world: &na::Projective3<f32>,
        L: Spectrum,
        w_light: na::Vector3<f32>,
    ) -> Self {
        Self {
            flags: LightFlags::DELTA_DIRECTION,
            light_to_world: *light_to_world,
            world_to_light: light_to_world.inverse(),
            L,
            w_light: (light_to_world * w_light).normalize(),
            world_center: na::Point3::origin(),
            world_radius: 0.0,
        }
    }
}

impl Light for DirectionalLight {
    fn sample_li(
        &self,
        reference: &Interaction,
        u: &na::Point2<f32>,
        wi: &mut na::Vector3<f32>,
        pdf: &mut f32,
        vis: &mut Option<VisibilityTester>,
    ) -> Spectrum {
        *wi = self.w_light;
        *pdf = 1.0;
        let p_outside = reference.p + self.w_light * (2.0 * self.world_radius);
        *vis = Some(VisibilityTester {
            p0: *reference,
            p1: Interaction {
                p: p_outside,
                time: reference.time,
                ..Default::default()
            },
        });

        self.L
    }

    fn power(&self) -> Spectrum {
        self.L * std::f32::consts::PI * self.world_radius * self.world_radius
    }

    fn pdf_li(&self, reference: &Interaction, wi: &na::Vector3<f32>) -> f32 {
        0.0
    }

    fn sample_le(
        &self,
        u1: &na::Point2<f32>,
        u2: &na::Point2<f32>,
        r: &mut Ray,
        n_light: &na::Vector3<f32>,
        pdf_pos: &mut f32,
        pdf_dir: &mut f32,
    ) {
        todo!()
    }

    fn pdf_le(&self, r: &Ray, n_light: &na::Vector3<f32>, pdf_pos: &mut f32, pdf_dir: &mut f32) {
        todo!()
    }

    fn preprocess(&mut self, world_bound: &Bounds3) {
        world_bound.bounding_sphere(&mut self.world_center, &mut self.world_radius);
    }
}

pub struct DiffuseAreaLight {
    l_emit: Spectrum,
    shape: Arc<dyn SyncShape>,
    area: f32,
}

impl DiffuseAreaLight {
    pub fn new(l_emit: Spectrum, shape: Arc<dyn SyncShape>) -> Self {
        Self {
            l_emit,
            area: shape.area(),
            shape,
        }
    }

    pub fn L(&self, inter: &Interaction, w: &na::Vector3<f32>) -> Spectrum {
        if inter.n.dot(&w) > 0.0 {
            self.l_emit
        } else {
            Spectrum::new(0.0)
        }
    }
}

impl Light for DiffuseAreaLight {
    fn sample_li(
        &self,
        reference: &Interaction,
        u: &nalgebra::Point2<f32>,
        wi: &mut nalgebra::Vector3<f32>,
        pdf: &mut f32,
        vis: &mut Option<VisibilityTester>,
    ) -> Spectrum {
        let p_shape = self.shape.sample_at_point(&reference, &u);

        *wi = (p_shape.p - reference.p).normalize();
        *pdf = self.shape.pdf_at_point(&reference, &wi);
        *vis = Some(VisibilityTester {
            p0: *reference,
            p1: p_shape,
        });

        self.L(&p_shape, &-*wi)
    }

    fn power(&self) -> Spectrum {
        todo!()
    }

    fn pdf_li(&self, reference: &Interaction, wi: &nalgebra::Vector3<f32>) -> f32 {
        self.shape.pdf_at_point(&reference, &wi)
    }

    fn sample_le(
        &self,
        u1: &nalgebra::Point2<f32>,
        u2: &nalgebra::Point2<f32>,
        r: &mut Ray,
        n_light: &nalgebra::Vector3<f32>,
        pdf_pos: &mut f32,
        pdf_dir: &mut f32,
    ) {
        todo!()
    }

    fn pdf_le(
        &self,
        r: &Ray,
        n_light: &nalgebra::Vector3<f32>,
        pdf_pos: &mut f32,
        pdf_dir: &mut f32,
    ) {
        todo!()
    }
}

pub struct InfiniteAreaLight {
    l_map: Box<MIPMap<Spectrum>>,
    world_center: na::Point3<f32>,
    world_radius: f32,
}
