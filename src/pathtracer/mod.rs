mod accelerator;
mod interaction;
mod light;
mod material;
mod primitive;
pub mod sampling;
mod shape;
mod texture;

use crate::common::bounds::{Bounds2i, Bounds3};
use crate::common::film::FilmTile;
use crate::common::ray::Ray;
use crate::common::spectrum::Spectrum;
use crate::common::{Camera, World};
use image::RgbImage;
use interaction::{Interaction, SurfaceInteraction};
use itertools::Itertools;
use light::{DirectionalLight, Light, PointLight, SyncLight};
use material::{BxDFType, Material, MatteMaterial, SyncMaterial};
use primitive::SyncPrimitive;
use rayon::prelude::*;
use shape::{shape_from_mesh, Shape};
use std::cell::RefCell;
use std::sync::Arc;
use std::{
    path::Path,
    time::{Duration, Instant},
};

pub enum TransportMode {
    Radiance,
    Importance,
}

#[derive(Debug)]
pub struct CameraSample {
    p_film: na::Point2<f32>,
}

impl Camera {
    pub fn generate_ray(&self, sample: &CameraSample) -> Ray {
        let cam_dir = self.cam_to_screen.unproject_point(
            &(self.raster_to_screen * na::Point3::new(sample.p_film.x, sample.p_film.y, 0.0)),
        );

        let cam_orig = na::Point3::<f32>::new(0.0, 0.0, 0.0);
        let world_orig = self.cam_to_world * cam_orig;
        let world_dir = self.cam_to_world * cam_dir.coords;
        Ray {
            o: world_orig,
            d: world_dir.normalize(),
            t_max: RefCell::new(f32::INFINITY),
        }
    }
}

pub struct RenderScene {
    scene: Box<dyn SyncPrimitive>,
    pub lights: Vec<Box<dyn SyncLight>>,
    materials: Vec<Arc<dyn SyncMaterial>>,
    world_bound: Bounds3,
}

impl RenderScene {
    pub fn from_world(world: &World) -> Self {
        let mut primitives: Vec<Arc<dyn SyncPrimitive>> = Vec::new();
        let materials = vec![Arc::new(MatteMaterial {}) as Arc<dyn SyncMaterial>];
        let mut lights = vec![Box::new(DirectionalLight::new(
            na::convert(na::Translation3::new(1.0, 3.5, 0.0)),
            Spectrum::new(1.0),
            na::Vector3::new(1.0, 1.0, 1.0),
        )) as Box<dyn SyncLight>];
        // let mut lights = vec![Box::new(PointLight::new(
        //     na::convert(na::Translation3::new(1.0, 3.5, 0.0)),
        //     Spectrum::new(10.0),
        // )) as Box<dyn SyncLight>];

        for obj in &world.objects {
            for shape in shape_from_mesh(&obj.mesh, &obj) {
                primitives.push(Arc::new(primitive::GeometricPrimitive {
                    shape: shape,
                    material: Arc::clone(&materials[0]),
                }) as Arc<dyn SyncPrimitive>)
            }
        }

        let bvh = Box::new(accelerator::BVH::new(primitives, &4)) as Box<dyn SyncPrimitive>;
        let world_bound = bvh.world_bound();

        for light in &mut lights {
            light.preprocess(&world_bound);
        }

        RenderScene {
            scene: bvh,
            materials,
            lights,
            world_bound,
        }
    }

    pub fn world_bound(&self) -> Bounds3 {
        self.world_bound
    }

    pub fn intersect<'a>(&'a self, r: &Ray, isect: &mut SurfaceInteraction<'a>) -> bool {
        self.scene.intersect(r, isect)
    }

    pub fn intersect_p(&self, r: &Ray) -> bool {
        self.scene.intersect_p(r)
    }
}

pub struct DirectLightingIntegrator {
    sampler: sampling::Sampler,
}

pub fn specular_reflect(r: &Ray, isect: &SurfaceInteraction, scene: &RenderScene, depth: u32) {
    let wo = isect.general.wo;
    // let wi = glm::zero();
    let pdf = 0.0;
    let bxdf_type = BxDFType::BSDF_REFLECTION | BxDFType::BSDF_SPECULAR;

    // let f = isect.bsdf.sam
}

impl DirectLightingIntegrator {
    pub fn new(sampler: sampling::Sampler) -> Self {
        DirectLightingIntegrator { sampler }
    }

    pub fn li(
        &self,
        r: &Ray,
        scene: &RenderScene,
        sampler: &mut sampling::Sampler,
        depth: u32,
    ) -> Spectrum {
        const MAX_DEPTH: u32 = 5;
        let mut L = Spectrum::new(0.0);
        let mut isect = Default::default();

        if !scene.intersect(&r, &mut isect) {
            for light in &scene.lights {
                L += light.le(&r);
            }
            return L;
        }
        trace!("intersected geometry at: {:?}", isect.general.p);

        let n = isect.shading.n;
        let wo = isect.general.wo;

        isect.compute_scattering_functions(r, TransportMode::Radiance);

        L += isect.le(&wo);

        for light in &scene.lights {
            let mut wi: na::Vector3<f32> = glm::zero();
            let mut pdf = 0.0;
            let mut visibility = None;
            let li = light.sample_li(
                &isect.general,
                &sampler.get_2d(),
                &mut wi,
                &mut pdf,
                &mut visibility,
            );
            trace!(
                "light {:p} gives li: {:?} for intersection point: {:?}",
                light,
                li,
                isect.general.p
            );

            if li.is_black() || pdf == 0.0 {
                continue;
            }

            let f = isect.bsdf.as_ref().unwrap().f(&wo, &wi, BxDFType::BSDF_ALL);
            trace!("bsdf f: {:?} for wo: {:?}, wi: {:?}", f, wo, wi);

            if !f.is_black() {
                if let Some(visibility) = visibility {
                    if visibility.unoccluded(&scene) {
                        trace!("light: {:p}, unoccluded", light);
                        L += f * li * wi.dot(&n) / pdf;
                    }
                }
            }
        }

        L
    }

    pub fn render_single_pixel(
        &self,
        camera: &mut Camera,
        pixel: na::Point2<i32>,
        scene: &RenderScene,
    ) {
        trace!("render single pixel: {:?}", pixel);
        trace!("camera at location: {:?}", camera.cam_to_world.translation);
        let mut pixel_sampler = self.sampler.clone_with_seed(0);
        pixel_sampler.start_pixel(&pixel);

        loop {
            let camera_sample = pixel_sampler.get_camera_sample(&pixel);
            trace!("generated camera sample: {:?}", camera_sample);
            let ray = camera.generate_ray(&camera_sample);
            trace!("generated ray: {:?}", ray);
            let mut L = Spectrum::new(0.0);
            L = self.li(&ray, &scene, &mut pixel_sampler, 0);
            trace!("output L: {:?}", L);

            if !pixel_sampler.start_next_sample() {
                break;
            }
        }

        trace!("actual image color: {:?}", camera.film.get_pixel(&pixel));
    }

    pub fn render(&self, camera: &mut Camera, scene: &RenderScene) {
        debug!(
            "start rendering image of size: {:?}",
            camera.film.get_sample_bounds().diagonal(),
        );
        let start = Instant::now();
        let sample_bounds = camera.film.get_sample_bounds();
        let sample_extent = sample_bounds.diagonal();
        const TILE_SIZE: i32 = 16;
        let num_tiles = na::Point2::new(
            (sample_extent.x + TILE_SIZE - 1) / TILE_SIZE,
            (sample_extent.y + TILE_SIZE - 1) / TILE_SIZE,
        );

        (0..num_tiles.x)
            .cartesian_product(0..num_tiles.y)
            .collect_vec()
            .par_iter()
            .map(|(x, y)| {
                let tile = na::Point2::new(*x, *y);
                let seed = (tile.y * num_tiles.x + tile.x) as u64;
                let mut tile_sampler = self.sampler.clone_with_seed(seed);

                let x0 = sample_bounds.p_min.x + tile.x * TILE_SIZE;
                let x1 = std::cmp::min(x0 + TILE_SIZE, sample_bounds.p_max.x);
                let y0 = sample_bounds.p_min.y + tile.y * TILE_SIZE;
                let y1 = std::cmp::min(y0 + TILE_SIZE, sample_bounds.p_max.y);

                let tile_bounds = Bounds2i {
                    p_min: na::Point2::new(x0, y0),
                    p_max: na::Point2::new(x1, y1),
                };
                let mut film_tile = camera.film.get_film_tile(&tile_bounds);

                for (x, y) in (tile_bounds.p_min.x..tile_bounds.p_max.x)
                    .cartesian_product(tile_bounds.p_min.y..tile_bounds.p_max.y)
                {
                    let pixel = na::Point2::new(x, y);
                    tile_sampler.start_pixel(&pixel);

                    loop {
                        let camera_sample = tile_sampler.get_camera_sample(&pixel);

                        let ray = camera.generate_ray(&camera_sample);

                        let mut L = Spectrum::new(0.0);
                        L = self.li(&ray, &scene, &mut tile_sampler, 0);

                        film_tile.add_sample(&camera_sample.p_film, &L);

                        if !tile_sampler.start_next_sample() {
                            break;
                        }
                    }
                }

                film_tile
            })
            .collect::<Vec<Box<FilmTile>>>()
            .drain(..)
            .for_each(|film_tile| {
                let film = &mut camera.film;
                film.merge_film_tile(film_tile);
            });

        let duration = start.elapsed();

        info!("rendering took: {:?}", duration);
    }
}
