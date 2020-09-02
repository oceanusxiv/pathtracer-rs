use super::{interaction::Interaction, RenderScene};
use crate::common::{bounds::Bounds3, ray::Ray, spectrum::Spectrum};
use ambassador::{delegatable_trait, Delegate};

bitflags! {
    pub struct LightFlags: u32 {
        const DELTA_POSITION = 1;
        const DELTA_DIRECTION = 2;
        const AREA = 4;
        const INFINITE = 8;
    }
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

#[delegatable_trait]
pub trait LightInterface {
    fn le(&self, r: &Ray) -> Spectrum {
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

    fn preprocess(&mut self, world_bound: &Bounds3) {}

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
}

#[derive(Delegate)]
#[delegate(LightInterface)]
pub enum Light {
    Point(PointLight),
    Directional(DirectionalLight),
}

pub struct PointLight {
    flags: LightFlags,
    num_samples: u32,
    light_to_world: na::Projective3<f32>,
    world_to_light: na::Projective3<f32>,
    p_light: na::Point3<f32>,
    I: Spectrum,
}

impl PointLight {
    pub fn new(light_to_world: na::Projective3<f32>, I: Spectrum) -> Self {
        PointLight {
            flags: LightFlags::DELTA_POSITION,
            num_samples: 1,
            light_to_world,
            world_to_light: light_to_world.inverse(),
            p_light: light_to_world * na::Point3::origin(),
            I,
        }
    }
}

impl LightInterface for PointLight {
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
        todo!()
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
        light_to_world: na::Projective3<f32>,
        L: Spectrum,
        w_light: na::Vector3<f32>,
    ) -> Self {
        DirectionalLight {
            flags: LightFlags::DELTA_POSITION,
            light_to_world,
            world_to_light: light_to_world.inverse(),
            L,
            w_light: (light_to_world * w_light).normalize(),
            world_center: na::Point3::origin(),
            world_radius: 0.0,
        }
    }
}

impl LightInterface for DirectionalLight {
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
        todo!()
    }

    fn preprocess(&mut self, world_bound: &Bounds3) {
        // debug!("scene world bound {:?}", scene.world_bound());
        world_bound.bounding_sphere(&mut self.world_center, &mut self.world_radius);
        // debug!("directional light world center: {:?}, radius: {:?}", self.world_center, self.world_radius);
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
