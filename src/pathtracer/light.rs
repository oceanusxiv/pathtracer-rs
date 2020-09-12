use std::{fs::File, io::BufReader, sync::Arc};

use super::{
    interaction::{Interaction, SurfaceMediumInteraction},
    sampling::Distribution2D,
    shape::SyncShape,
    texture::{MIPMap, SyncTexture},
    RenderScene,
};
use crate::common::{
    bounds::Bounds3,
    math::spherical_phi,
    math::spherical_theta,
    math::INV_2_PI,
    ray::{Ray, RayDifferential},
    spectrum::Spectrum,
    WrapMode,
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
    light_to_world: na::Projective3<f32>,
    world_to_light: na::Projective3<f32>,
    world_center: na::Point3<f32>,
    world_radius: f32,
    distribution: Box<Distribution2D>,
    log: slog::Logger,
}

fn read_hdr_image_to_mat(path: &str, l: Spectrum) -> anyhow::Result<na::DMatrix<Spectrum>> {
    let file = File::open(path)?;
    let file = BufReader::new(file);
    let decoder = image::hdr::HDRDecoder::new(file)?;
    let metadata = decoder.metadata();
    let image = decoder.read_image_hdr()?;
    Ok(na::DMatrix::from_fn(
        metadata.height as usize,
        metadata.width as usize,
        |row, col| {
            let rgb = &image[row * metadata.width as usize + col];

            l * Spectrum {
                r: rgb[0],
                g: rgb[1],
                b: rgb[2],
            }
        },
    ))
}

impl InfiniteAreaLight {
    pub fn new(
        log: &slog::Logger,
        light_to_world: na::Projective3<f32>,
        l: Spectrum,
        hdr_map_path: &str,
    ) -> Self {
        let log = log.new(o!());
        let mut texels: Option<na::DMatrix<Spectrum>> = None;
        if !hdr_map_path.is_empty() {
            match read_hdr_image_to_mat(hdr_map_path, l) {
                Ok(mat) => texels = Some(mat),
                Err(error) => {
                    error!(
                        log,
                        "error reading hdr image, falling back to black. Error: {:?}", error
                    );
                }
            }
        }

        let texels = if let Some(texels) = texels {
            texels
        } else {
            na::DMatrix::<Spectrum>::from_element(1, 1, l)
        };

        let width = 2 * texels.ncols();
        let height = 2 * texels.nrows();
        let f_width = 0.5 / width.min(height) as f32;
        let mut img = Vec::with_capacity(width * height);
        let l_map = Box::new(MIPMap::new(&log, texels, false, WrapMode::Repeat));
        for v in 0..height {
            let vp = (v as f32 + 0.5) / (height as f32);
            let sin_theta = (std::f32::consts::PI * vp).sin();
            for u in 0..width {
                let up = (u as f32 + 0.5) / (width as f32);
                img.push(sin_theta * l_map.lookup_width(&na::Point2::new(up, vp), f_width).y());
            }
        }

        Self {
            l_map,
            light_to_world,
            world_to_light: light_to_world.inverse(),
            world_center: na::Point3::origin(),
            world_radius: 0.0,
            distribution: Box::new(Distribution2D::new(&img[..], width, height)),
            log,
        }
    }
}

impl Light for InfiniteAreaLight {
    fn sample_li(
        &self,
        reference: &Interaction,
        u: &nalgebra::Point2<f32>,
        wi: &mut nalgebra::Vector3<f32>,
        pdf: &mut f32,
        vis: &mut Option<VisibilityTester>,
    ) -> Spectrum {
        let mut map_pdf = 0.0;
        let uv = self.distribution.sample_continuous(&u, &mut map_pdf);
        if map_pdf == 0.0 {
            return Spectrum::new(0.0);
        }

        let theta = uv[1] * std::f32::consts::PI;
        let phi = uv[0] * 2.0 * std::f32::consts::PI;
        let cos_theta = theta.cos();
        let sin_theta = theta.sin();
        let sin_phi = phi.sin();
        let cos_phi = phi.cos();

        *wi = self.light_to_world
            * na::Vector3::new(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta);
        if sin_theta == 0.0 {
            *pdf = 0.0;
        } else {
            *pdf = map_pdf / (2.0 * std::f32::consts::PI * std::f32::consts::PI * sin_theta);
        }

        *vis = Some(VisibilityTester {
            p0: *reference,
            p1: Interaction {
                p: reference.p + *wi * (2.0 * self.world_radius),
                time: reference.time,
                ..Default::default()
            },
        });

        self.l_map.lookup_width(&uv, 0.0)
    }

    fn power(&self) -> Spectrum {
        todo!()
    }

    fn pdf_li(&self, _reference: &Interaction, w: &nalgebra::Vector3<f32>) -> f32 {
        let wi = self.world_to_light * w;
        let theta = spherical_theta(&wi);
        let phi = spherical_phi(&wi);
        let sin_theta = theta.sin();
        if sin_theta == 0.0 {
            0.0
        } else {
            self.distribution.pdf(&na::Point2::new(
                phi * INV_2_PI,
                theta * std::f32::consts::FRAC_1_PI,
            )) / (2.0 * std::f32::consts::PI * std::f32::consts::PI * sin_theta)
        }
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

    fn preprocess(&mut self, world_bound: &Bounds3) {
        world_bound.bounding_sphere(&mut self.world_center, &mut self.world_radius);
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

    fn le(&self, r: &RayDifferential) -> Spectrum {
        let w = (self.world_to_light * r.ray.d).normalize();
        let st = na::Point2::new(
            spherical_phi(&w) * INV_2_PI,
            spherical_theta(&w) * std::f32::consts::FRAC_1_PI,
        );

        trace!(self.log, "lookup env map with st: {:?}", st);

        self.l_map.lookup_width(&st, 0.0)
    }

    fn flags(&self) -> LightFlags {
        LightFlags::INFINITE
    }
}
