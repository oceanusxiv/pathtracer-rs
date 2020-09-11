use std::sync::Arc;

use super::{
    interaction::{Interaction, SurfaceMediumInteraction},
    shape::SyncShape,
    texture::{MIPMap, SyncTexture},
    RenderScene,
};
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

    fn preprocess(&mut self, _world_bound: &Bounds3) {}

    fn get_num_samples(&self) -> usize {
        1
    }

    fn flags(&self) -> LightFlags;
}

pub trait SyncLight: Light + Send + Sync {}
impl<T> SyncLight for T where T: Light + Send + Sync {}

pub struct PointLight {
    p_light: na::Point3<f32>,
    i: Spectrum,
}

impl PointLight {
    pub fn new(light_to_world: &na::Projective3<f32>, i: Spectrum) -> Self {
        Self {
            p_light: light_to_world * na::Point3::origin(),
            i,
        }
    }
}

impl Light for PointLight {
    fn sample_li(
        &self,
        reference: &Interaction,
        _u: &na::Point2<f32>,
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

        self.i / (self.p_light - reference.p).norm_squared()
    }

    fn power(&self) -> Spectrum {
        4.0 * std::f32::consts::PI * self.i
    }

    fn pdf_li(&self, _reference: &Interaction, _wi: &na::Vector3<f32>) -> f32 {
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

    fn flags(&self) -> LightFlags {
        LightFlags::DELTA_POSITION
    }
}

pub struct DirectionalLight {
    l: Spectrum,
    w_light: na::Vector3<f32>,
    world_center: na::Point3<f32>,
    world_radius: f32,
}

impl DirectionalLight {
    pub fn new(
        light_to_world: &na::Projective3<f32>,
        l: Spectrum,
        w_light: na::Vector3<f32>,
    ) -> Self {
        Self {
            l,
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
        _u: &na::Point2<f32>,
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

        self.l
    }

    fn power(&self) -> Spectrum {
        self.l * std::f32::consts::PI * self.world_radius * self.world_radius
    }

    fn pdf_li(&self, _reference: &Interaction, _wi: &na::Vector3<f32>) -> f32 {
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

    fn flags(&self) -> LightFlags {
        LightFlags::DELTA_DIRECTION
    }
}

pub struct DiffuseAreaLight {
    ke: Arc<dyn SyncTexture<Spectrum>>,
    shape: Arc<dyn SyncShape>,
    num_samples: usize,
    area: f32,
}

impl DiffuseAreaLight {
    pub fn new(
        ke: Arc<dyn SyncTexture<Spectrum>>,
        shape: Arc<dyn SyncShape>,
        num_samples: usize,
    ) -> Self {
        Self {
            ke,
            area: shape.area(),
            num_samples,
            shape,
        }
    }

    pub fn l(&self, inter: &SurfaceMediumInteraction, w: &na::Vector3<f32>) -> Spectrum {
        if inter.general.n.dot(&w) > 0.0 {
            self.ke.evaluate(&inter)
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

        *wi = (p_shape.general.p - reference.p).normalize();
        *pdf = self.shape.pdf_at_point(&reference, &wi);
        *vis = Some(VisibilityTester {
            p0: *reference,
            p1: p_shape.general,
        });

        self.l(&p_shape, &-*wi)
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

    fn get_num_samples(&self) -> usize {
        self.num_samples
    }

    fn flags(&self) -> LightFlags {
        LightFlags::AREA
    }
}

pub struct InfiniteAreaLight {
    l_map: Box<MIPMap<Spectrum>>,
    world_center: na::Point3<f32>,
    world_radius: f32,
}
