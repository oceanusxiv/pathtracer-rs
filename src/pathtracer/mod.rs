mod accelerator;
mod interaction;
mod light;
mod material;
mod primitive;
mod sampling;
mod shape;

use crate::common::bounds::Bounds2i;
use crate::common::film::FilmTile;
use crate::common::ray::Ray;
use crate::common::spectrum::Spectrum;
use crate::common::{Camera, World};
use image::RgbImage;
use indicatif::ParallelProgressIterator;
use interaction::{Interaction, SurfaceInteraction};
use itertools::Itertools;
use light::{Light, PointLight, SyncLight};
use material::{BxDFType, Material, MatteMaterial, SyncMaterial};
use primitive::SyncPrimitive;
use rayon::prelude::*;
use shape::Shape;
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
pub trait Sampler {}

impl Camera {
    pub fn generate_ray(&self, film_point: &na::Point2<f32>) -> Ray {
        let cam_dir = self.cam_to_screen.unproject_point(
            &(self.raster_to_screen * na::Point3::new(film_point.x, film_point.y, 0.0)),
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
    materials: Vec<Arc<dyn SyncMaterial>>,
    pub lights: Vec<Box<dyn SyncLight>>,
}

impl RenderScene {
    pub fn from_world(world: &World) -> Self {
        let mut primitives: Vec<Arc<dyn SyncPrimitive>> = Vec::new();
        let materials = vec![Arc::new(MatteMaterial {}) as Arc<dyn SyncMaterial>];
        let lights = vec![Box::new(PointLight::new(
            na::convert(na::Translation3::new(0.0, 3.0, 0.0)),
            Spectrum::new(1.0),
        )) as Box<dyn SyncLight>];

        for obj in &world.objects {
            for shape in obj.mesh.to_shapes(&obj) {
                primitives.push(Arc::new(primitive::GeometricPrimitive {
                    shape: shape,
                    material: Arc::clone(&materials[0]),
                }) as Arc<dyn SyncPrimitive>)
            }
        }

        RenderScene {
            scene: Box::new(accelerator::BVH::new(primitives, &4)) as Box<dyn SyncPrimitive>,
            materials,
            lights,
        }
    }

    pub fn intersect<'a>(&'a self, r: &Ray, isect: &mut SurfaceInteraction<'a>) -> bool {
        self.scene.intersect(r, isect)
    }

    pub fn intersect_p(&self, r: &Ray) -> bool {
        self.scene.intersect_p(r)
    }
}

pub trait Integrator {}

pub struct DirectLightingIntegrator {}

impl DirectLightingIntegrator {
    pub fn new() -> Self {
        DirectLightingIntegrator {}
    }

    pub fn li(&self, r: &Ray, scene: &RenderScene, depth: u32) -> Spectrum {
        const MAX_DEPTH: u32 = 5;
        let mut L = Spectrum::new(0.0);
        let mut isect = Default::default();

        if !scene.intersect(&r, &mut isect) {
            for light in &scene.lights {
                L += light.le(&r);
            }
            return L;
        }

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
                &na::Point2::new(0.0, 0.0),
                &mut wi,
                &mut pdf,
                &mut visibility,
            );

            if li.is_black() || pdf == 0.0 {
                continue;
            }

            let f = isect.bsdf.as_ref().unwrap().f(&wo, &wi, BxDFType::BSDF_ALL);

            if !f.is_black() {
                if let Some(visibility) = visibility {
                    if visibility.unoccluded(&scene) {
                        L += f * li * wi.dot(&n) / pdf;
                    }
                }
            }
        }

        L
    }

    pub fn render(&self, camera: &mut Camera, scene: &RenderScene, out_path: &Path) {
        println!(
            "start rendering image of size: {:?}",
            camera.film.get_sample_bounds(),
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
            .progress()
            .map(|(x, y)| {
                let tile = na::Point2::new(*x, *y);
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
                    let film_point = na::Point2::new(x as f32, y as f32) + glm::vec2(0.5, 0.5);
                    let ray = camera.generate_ray(&film_point);

                    let mut L = Spectrum::new(0.0);
                    L = self.li(&ray, &scene, 0);

                    film_tile.add_sample(&film_point, &L);
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

        println!("rendering took: {:?}", duration);
        println!("saving image to {:?}", out_path);
        camera.film.save(out_path);
    }
}

impl Integrator for DirectLightingIntegrator {}
