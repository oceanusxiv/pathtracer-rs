mod accelerator;
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
use itertools::Itertools;
use material::Material;
use rayon::prelude::*;
use shape::Shape;
use std::cell::RefCell;
use std::sync::Arc;
use std::time::{Duration, Instant};

pub struct SurfaceInteraction {
    pub p: glm::Vec3,
    pub p_error: glm::Vec3,
    pub wo: glm::Vec3,
    pub n: glm::Vec3,
}

impl SurfaceInteraction {
    pub fn new() -> Self {
        SurfaceInteraction {
            p: glm::zero(),
            p_error: glm::zero(),
            wo: glm::zero(),
            n: glm::zero(),
        }
    }
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
            t_max: RefCell::new(std::f32::INFINITY),
        }
    }
}

pub struct RenderScene {
    scene: Box<dyn primitive::Primitive + Send + Sync>,
}

impl RenderScene {
    pub fn from_world(world: &World) -> Self {
        let mut primitives: Vec<Arc<dyn primitive::Primitive + Send + Sync>> = Vec::new();

        for obj in &world.objects {
            for shape in obj.mesh.to_shapes(&obj) {
                primitives.push(Arc::new(primitive::GeometricPrimitive { shape: shape })
                    as Arc<dyn primitive::Primitive + Send + Sync>)
            }
        }

        RenderScene {
            scene: Box::new(accelerator::BVH::new(primitives, &4))
                as Box<dyn primitive::Primitive + Send + Sync>,
        }
    }
}

pub trait Integrator {}

pub struct DirectLightingIntegrator {}

impl DirectLightingIntegrator {
    pub fn new() -> Self {
        DirectLightingIntegrator {}
    }

    pub fn render(&self, camera: &mut Camera, scene: &RenderScene, out_path: &str) {
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
                    let mut isect = SurfaceInteraction::new();
                    if scene.scene.intersect(&ray, &mut isect) {
                        film_tile.add_sample(
                            &film_point,
                            &Spectrum {
                                r: 1.0,
                                g: 1.0,
                                b: 1.0,
                            },
                        );
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

        println!("rendering took: {:?}", duration);
        println!("saving image to {:?}", out_path);
        camera.film.save(out_path);
    }
}

impl Integrator for DirectLightingIntegrator {}
