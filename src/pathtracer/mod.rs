mod accelerator;
mod bsdf;
mod bxdf;
mod interaction;
pub mod light;
mod material;
mod primitive;
pub mod sampling;
mod shape;
mod texture;

use crate::common::bounds::{Bounds2i, Bounds3};
use crate::common::film::FilmTile;
use crate::common::ray::{Ray, RayDifferential};
use crate::common::spectrum::Spectrum;
use crate::common::{Camera, World};
use bxdf::BxDFType;
use indicatif::ParallelProgressIterator;
use interaction::SurfaceInteraction;
use itertools::Itertools;
use light::{DiffuseAreaLight, DirectionalLight, Light, PointLight, SyncLight};
use material::{Material, MaterialInterface};
use primitive::SyncPrimitive;
use rayon::prelude::*;
use shape::shape_from_mesh;
use std::cell::RefCell;
use std::sync::Arc;
use std::time::Instant;

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
        let p_camera = self.cam_to_screen.unproject_point(
            &(self.raster_to_screen * na::Point3::new(sample.p_film.x, sample.p_film.y, 0.0)),
        );

        let cam_orig = na::Point3::<f32>::new(0.0, 0.0, 0.0);
        let world_orig = self.cam_to_world * cam_orig;
        let world_dir = self.cam_to_world * p_camera.coords;
        Ray {
            o: world_orig,
            d: world_dir.normalize(),
            t_max: RefCell::new(f32::INFINITY),
        }
    }

    pub fn generate_ray_differential(&self, sample: &CameraSample) -> RayDifferential {
        let p_camera = self.cam_to_screen.unproject_point(
            &(self.raster_to_screen * na::Point3::new(sample.p_film.x, sample.p_film.y, 0.0)),
        );

        let cam_orig = na::Point3::<f32>::new(0.0, 0.0, 0.0);
        let world_orig = self.cam_to_world * cam_orig;
        let world_dir = self.cam_to_world * p_camera.coords;
        let rx_world_dir = self.cam_to_world * (p_camera.coords + self.dx_camera);
        let ry_world_dir = self.cam_to_world * (p_camera.coords + self.dy_camera);
        RayDifferential {
            ray: Ray {
                o: world_orig,
                d: world_dir.normalize(),
                t_max: RefCell::new(f32::INFINITY),
            },
            has_differentials: true,
            rx_origin: world_orig,
            ry_origin: world_orig,
            rx_direction: rx_world_dir.normalize(),
            ry_direction: ry_world_dir.normalize(),
        }
    }
}

pub struct RenderScene {
    scene: Box<dyn SyncPrimitive>,
    pub lights: Vec<Arc<dyn SyncLight>>,
    materials: Vec<Arc<Material>>,
    world_bound: Bounds3,
}

impl RenderScene {
    pub fn from_world(log: &slog::Logger, world: &World) -> Self {
        let mut primitives: Vec<Arc<dyn SyncPrimitive>> = Vec::new();
        let mut materials = Vec::new();
        let mut lights: Vec<Arc<dyn SyncLight>> = Vec::new();

        for mat in &world.materials {
            materials.push(Arc::new(Material::from_gltf(log, &**mat)));
        }

        for obj in &world.objects {
            for shape in shape_from_mesh(
                log,
                &obj.mesh,
                &obj,
                world.materials[obj.material.index]
                    .pbr_metallic_roughness
                    .alpha_texture
                    .as_ref(),
            ) {
                let emissive_factor = obj.material.emissive_factor;
                let some_area_light;
                // only create area light if object material is emissive
                if emissive_factor[0] > 0.0 && emissive_factor[1] > 0.0 && emissive_factor[2] > 0.0
                {
                    let area_light = Arc::new(DiffuseAreaLight::new());
                    lights.push(Arc::clone(&area_light) as Arc<dyn SyncLight>);
                    some_area_light = Some(Arc::clone(&area_light));
                } else {
                    some_area_light = None;
                }

                primitives.push(Arc::new(primitive::GeometricPrimitive::new(
                    shape,
                    Arc::clone(&materials[obj.material.index]),
                    some_area_light,
                )) as Arc<dyn SyncPrimitive>)
            }
        }

        let bvh = Box::new(accelerator::BVH::new(log, primitives, &4)) as Box<dyn SyncPrimitive>;
        let world_bound = bvh.world_bound();

        for light_info in &world.lights {
            lights.push(Light::from_gltf(light_info, &world_bound));
        }

        if lights.is_empty() {
            let default_direction_light = Arc::new(DirectionalLight::new(
                na::convert(na::Translation3::new(1.0, 3.5, 0.0)),
                Spectrum::new(10.0),
                na::Vector3::new(0.0, 1.0, 0.5),
                &world_bound,
            ));
            lights.push(default_direction_light as Arc<dyn SyncLight>);
            let default_point_light = Arc::new(PointLight::new(
                na::convert(na::Translation3::new(1.0, 3.5, 0.0)),
                Spectrum::new(30.0),
            ));
            lights.push(default_point_light as Arc<dyn SyncLight>);
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

pub enum LightStrategy {
    UniformSampleAll,
    UniformSampleOne,
}

pub struct DirectLightingIntegrator {
    sampler: sampling::Sampler,
    log: slog::Logger,
}

impl DirectLightingIntegrator {
    pub fn new(log: &slog::Logger, sampler: sampling::Sampler) -> Self {
        let log = log.new(o!("integrator" => "direct lighting integrator"));
        Self { sampler, log }
    }

    pub fn set_sampler(&mut self, sampler: sampling::Sampler) {
        self.sampler = sampler;
    }

    pub fn preprocess() {}

    fn specular_reflect(
        &self,
        r: &RayDifferential,
        isect: &SurfaceInteraction,
        scene: &RenderScene,
        mut sampler: &mut sampling::Sampler,
        depth: u32,
    ) -> Spectrum {
        let wo = isect.general.wo;
        let mut wi = glm::zero();
        let mut pdf = 0.0;
        let bxdf_type = BxDFType::BSDF_REFLECTION | BxDFType::BSDF_SPECULAR;

        let f = isect.bsdf.as_ref().unwrap().sample_f(
            &wo,
            &mut wi,
            &sampler.get_2d(),
            &mut pdf,
            bxdf_type,
            &mut None,
        );

        let ns = isect.shading.n;
        if pdf > 0.0 && !f.is_black() && wi.dot(&ns).abs() != 0.0 {
            // Compute ray differential rd for specular reflection
            let mut rd = RayDifferential::new(isect.general.spawn_ray(&wi));
            if r.has_differentials {
                rd.has_differentials = true;
                rd.rx_origin = isect.general.p + isect.dpdx;
                rd.ry_origin = isect.general.p + isect.dpdy;
                let dndx = isect.shading.dndu * isect.dudx + isect.shading.dndv * isect.dvdx;
                let dndy = isect.shading.dndu * isect.dudy + isect.shading.dndv * isect.dvdy;
                let dwodx = -r.rx_direction - wo;
                let dwody = -r.ry_direction - wo;
                let dDNdx = dwodx.dot(&ns) + wo.dot(&dndx);
                let dDNdy = dwody.dot(&ns) + wo.dot(&dndy);
                rd.rx_direction = wi - dwodx + 2.0 * (wo.dot(&ns) * dndx + dDNdx * ns);
                rd.ry_direction = wi - dwody + 2.0 * (wo.dot(&ns) * dndy + dDNdy * ns);
            }
            f * self.li(&rd, &scene, &mut sampler, depth + 1) * wi.dot(&ns).abs() / pdf
        } else {
            Spectrum::new(0.0)
        }
    }

    fn specular_transmit(
        &self,
        r: &RayDifferential,
        isect: &SurfaceInteraction,
        scene: &RenderScene,
        mut sampler: &mut sampling::Sampler,
        depth: u32,
    ) -> Spectrum {
        let wo = isect.general.wo;
        let mut wi = glm::zero();
        let mut pdf = 0.0;
        let p = isect.general.p;
        let mut ns = isect.shading.n;
        let bsdf = isect.bsdf.as_ref().unwrap();
        let f = bsdf.sample_f(
            &wo,
            &mut wi,
            &sampler.get_2d(),
            &mut pdf,
            BxDFType::BSDF_TRANSMISSION | BxDFType::BSDF_SPECULAR,
            &mut None,
        );
        let mut L = Spectrum::new(0.0);

        if pdf > 0.0 && !f.is_black() && wi.dot(&ns).abs() != 0.0 {
            let mut rd = RayDifferential::new(isect.general.spawn_ray(&wi));
            if rd.has_differentials {
                // Compute ray differential rd for specular transmission
                rd.has_differentials = true;
                rd.rx_origin = isect.general.p + isect.dpdx;
                rd.ry_origin = isect.general.p + isect.dpdy;
                let mut dndx = isect.shading.dndu * isect.dudx + isect.shading.dndv * isect.dvdx;
                let mut dndy = isect.shading.dndu * isect.dudy + isect.shading.dndv * isect.dvdy;

                let mut eta = 1.0 / bsdf.eta;
                if wo.dot(&ns) < 0.0 {
                    eta = 1.0 / eta;
                    ns = -ns;
                    dndx = -dndx;
                    dndy = -dndy;
                }

                let dwodx = -r.rx_direction - wo;
                let dwody = -r.ry_direction - wo;
                let dDNdx = dwodx.dot(&ns) + wo.dot(&dndx);
                let dDNdy = dwody.dot(&ns) + wo.dot(&dndy);

                let mu = eta * wo.dot(&ns) - wi.dot(&ns).abs();
                let dmudx = (eta - (eta * eta * wo.dot(&ns)) / wi.dot(&ns).abs()) * dDNdx;
                let dmudy = (eta - (eta * eta * wo.dot(&ns)) / wi.dot(&ns).abs()) * dDNdy;

                rd.rx_direction = wi - eta * dwodx + (mu * dndx + dmudx * ns);
                rd.ry_direction = wi - eta * dwody + (mu * dndy + dmudy * ns);
            }
            L = f * self.li(&rd, &scene, &mut sampler, depth + 1) * wi.dot(&ns).abs() / pdf
        }

        L
    }

    fn li(
        &self,
        r: &RayDifferential,
        scene: &RenderScene,
        mut sampler: &mut sampling::Sampler,
        depth: u32,
    ) -> Spectrum {
        const MAX_DEPTH: u32 = 5;
        let mut L = Spectrum::new(0.0);
        let mut isect = Default::default();

        if !scene.intersect(&r.ray, &mut isect) {
            for light in &scene.lights {
                L += light.le(&r);
            }
            return L;
        }
        trace!(self.log, "intersected geometry at: {:?}", isect.general.p);

        isect.compute_scattering_functions(r, TransportMode::Radiance);

        let n = isect.shading.n;
        let wo = isect.general.wo;

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
                self.log,
                "light {:p} gives li: {:?} for intersection point: {:?}",
                light,
                li,
                isect.general.p
            );

            if li.is_black() || pdf == 0.0 {
                continue;
            }

            let f = isect.bsdf.as_ref().unwrap().f(&wo, &wi, BxDFType::BSDF_ALL);
            trace!(self.log, "bsdf f: {:?} for wo: {:?}, wi: {:?}", f, wo, wi);

            if !f.is_black() {
                if let Some(visibility) = visibility {
                    if visibility.unoccluded(&scene) {
                        trace!(self.log, "light: {:p}, unoccluded", light);
                        L += f * li * wi.dot(&n) / pdf;
                    }
                }
            }
        }

        if depth + 1 < MAX_DEPTH {
            L += self.specular_reflect(&r, &isect, &scene, &mut sampler, depth);
            L += self.specular_transmit(&r, &isect, &scene, &mut sampler, depth);
        }

        L
    }

    pub fn render_single_pixel(
        &self,
        camera: &mut Camera,
        pixel: na::Point2<i32>,
        scene: &RenderScene,
    ) {
        trace!(self.log, "render single pixel: {:?}", pixel);
        trace!(
            self.log,
            "camera at location: {:?}",
            camera.cam_to_world.translation
        );
        let mut pixel_sampler = self.sampler.clone_with_seed(0);
        pixel_sampler.start_pixel(&pixel);

        loop {
            let camera_sample = pixel_sampler.get_camera_sample(&pixel);
            trace!(self.log, "generated camera sample: {:?}", camera_sample);
            let mut ray = camera.generate_ray_differential(&camera_sample);
            ray.scale_differentials(1.0 / (pixel_sampler.samples_per_pixel() as f32).sqrt());
            trace!(self.log, "generated ray: {:?}", ray);
            let mut L = Spectrum::new(0.0);
            L = self.li(&ray, &scene, &mut pixel_sampler, 0);
            trace!(self.log, "output L: {:?}", L);

            if !pixel_sampler.start_next_sample() {
                break;
            }
        }

        trace!(
            self.log,
            "actual image color: {:?}",
            camera.film.get_pixel(&pixel)
        );
    }

    pub fn render(&self, camera: &mut Camera, scene: &RenderScene) {
        debug!(
            self.log,
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
            .progress_count((num_tiles.x * num_tiles.y) as u64)
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

                        let mut ray = camera.generate_ray_differential(&camera_sample);
                        ray.scale_differentials(
                            1.0 / (tile_sampler.samples_per_pixel() as f32).sqrt(),
                        );

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

        info!(self.log, "rendering took: {:?}", duration);
    }
}
