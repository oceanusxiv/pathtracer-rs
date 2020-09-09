use super::bxdf::BxDFType;
use super::interaction::{Interaction, SurfaceMediumInteraction};
use super::sampling::{Sampler, SamplerBuilder};
use super::{light::SyncLight, RenderScene, TransportMode};
use crate::common::bounds::Bounds2i;
use crate::common::film::FilmTile;
use crate::common::ray::RayDifferential;
use crate::common::spectrum::Spectrum;
use crate::common::Camera;
use indicatif::ParallelProgressIterator;
use itertools::Itertools;
use rayon::prelude::*;
use std::time::Instant;

#[derive(Debug, Eq, PartialEq)]
pub enum LightStrategy {
    UniformSampleAll,
    UniformSampleOne,
}

fn estimate_direct(
    it: &SurfaceMediumInteraction,
    u_scattering: &na::Point2<f32>,
    light: &dyn SyncLight,
    u_light: &na::Point2<f32>,
    scene: &RenderScene,
    sampler: &Sampler,
    specular: bool,
) -> Spectrum {
    let bsdf_flags = if specular {
        BxDFType::BSDF_ALL
    } else {
        BxDFType::BSDF_ALL - BxDFType::BSDF_SPECULAR
    };
    let mut ld = Spectrum::new(0.0);

    let mut wi = na::Vector3::zeros();
    let mut light_pdf = 0.0;
    let scattering_pdf = 0.0;
    let mut visibility = None;
    let li = light.sample_li(
        &it.general,
        &u_light,
        &mut wi,
        &mut light_pdf,
        &mut visibility,
    );
    if light_pdf > 0.0 && !li.is_black() {}

    ld
}

fn uniform_sample_all_lights(
    it: &SurfaceMediumInteraction,
    scene: &RenderScene,
    sampler: &mut Sampler,
    num_light_samples: Vec<usize>,
) -> Spectrum {
    let mut l = Spectrum::new(0.0);

    for j in 0..scene.lights.len() {
        let light = scene.lights[j].as_ref();
        let num_samples = num_light_samples[j];
        let u_light_array = sampler.get_2d_array(num_samples);
        let u_scattering_array = sampler.get_2d_array(num_samples);

        if u_light_array.is_none() || u_scattering_array.is_none() {
            let u_light = sampler.get_2d();
            let u_scattering = sampler.get_2d();
            l += estimate_direct(&it, &u_scattering, light, &u_light, &scene, &sampler, false);
        } else {
            let mut ld = Spectrum::new(0.0);
            let u_scattering_array = u_scattering_array.unwrap();
            let u_light_array = u_light_array.unwrap();
            for k in 0..num_samples {
                ld += estimate_direct(
                    &it,
                    &u_scattering_array[k],
                    light,
                    &u_light_array[k],
                    &scene,
                    &sampler,
                    false,
                );
            }
            l += ld / num_samples as f32;
        }
    }

    l
}

fn uniform_sample_one_light(
    it: &SurfaceMediumInteraction,
    scene: &RenderScene,
    sampler: &mut Sampler,
) -> Spectrum {
    let num_lights = scene.lights.len();
    if num_lights == 0 {
        return Spectrum::new(0.0);
    }

    let u_light = sampler.get_2d();
    let u_scattering = sampler.get_2d();
    let light_idx = ((sampler.get_1d() * num_lights as f32).floor() as usize).min(num_lights - 1);
    let light = scene.lights[light_idx].as_ref();
    num_lights as f32
        * estimate_direct(&it, &u_scattering, light, &u_light, &scene, &sampler, false)
}

pub struct DirectLightingIntegrator {
    sampler_builder: SamplerBuilder,
    max_depth: u32,
    strategy: LightStrategy,
    num_light_samples: Vec<usize>,
    log: slog::Logger,
}

impl DirectLightingIntegrator {
    pub fn new(log: &slog::Logger, sampler_builder: SamplerBuilder, max_depth: u32) -> Self {
        let log = log.new(o!("module" => "integrator"));
        Self {
            sampler_builder,
            max_depth,
            strategy: LightStrategy::UniformSampleAll,
            num_light_samples: vec![],
            log,
        }
    }

    // this should be run once per scene change or sampler change
    // NOTE: sampler should be reset every scene change as well
    pub fn preprocess(&mut self, scene: &RenderScene) {
        self.num_light_samples.clear();

        if self.strategy == LightStrategy::UniformSampleAll {
            for light in &scene.lights {
                self.num_light_samples.push(light.get_num_samples());
            }

            for _ in 0..self.max_depth {
                for j in 0..scene.lights.len() {
                    self.sampler_builder
                        .request_2d_array(self.num_light_samples[j]);
                    self.sampler_builder
                        .request_2d_array(self.num_light_samples[j]);
                }
            }
        }
    }

    fn specular_reflect(
        &self,
        r: &RayDifferential,
        isect: &SurfaceMediumInteraction,
        scene: &RenderScene,
        mut sampler: &mut Sampler,
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
        let l;
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
                let d_dndx = dwodx.dot(&ns) + wo.dot(&dndx);
                let d_dndy = dwody.dot(&ns) + wo.dot(&dndy);
                rd.rx_direction = wi - dwodx + 2.0 * (wo.dot(&ns) * dndx + d_dndx * ns);
                rd.ry_direction = wi - dwody + 2.0 * (wo.dot(&ns) * dndy + d_dndy * ns);
            }
            l = f * self.li(&rd, &scene, &mut sampler, depth + 1) * wi.dot(&ns).abs() / pdf;
        } else {
            l = Spectrum::new(0.0);
        }

        trace!(
            self.log,
            "L: {:?}, after specular reflect depth: {:?}",
            l,
            depth
        );

        l
    }

    fn specular_transmit(
        &self,
        r: &RayDifferential,
        isect: &SurfaceMediumInteraction,
        scene: &RenderScene,
        mut sampler: &mut Sampler,
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
        let mut l = Spectrum::new(0.0);

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
                let d_dndx = dwodx.dot(&ns) + wo.dot(&dndx);
                let d_dndy = dwody.dot(&ns) + wo.dot(&dndy);

                let mu = eta * wo.dot(&ns) - wi.dot(&ns).abs();
                let dmudx = (eta - (eta * eta * wo.dot(&ns)) / wi.dot(&ns).abs()) * d_dndx;
                let dmudy = (eta - (eta * eta * wo.dot(&ns)) / wi.dot(&ns).abs()) * d_dndy;

                rd.rx_direction = wi - eta * dwodx + (mu * dndx + dmudx * ns);
                rd.ry_direction = wi - eta * dwody + (mu * dndy + dmudy * ns);
            }
            l = f * self.li(&rd, &scene, &mut sampler, depth + 1) * wi.dot(&ns).abs() / pdf
        }

        trace!(
            self.log,
            "L: {:?}, after specular transmit depth: {:?}, pdf: {:?}, f: {:?}, wi: {:?}, ns: {:?}",
            l,
            depth,
            pdf,
            f,
            wi,
            ns
        );

        l
    }

    fn li(
        &self,
        ray: &RayDifferential,
        scene: &RenderScene,
        mut sampler: &mut Sampler,
        depth: u32,
    ) -> Spectrum {
        let mut l = Spectrum::new(0.0);
        let mut isect = Default::default();

        if !scene.intersect(&ray.ray, &mut isect) {
            for light in &scene.lights {
                l += light.le(&ray);
            }
            return l;
        }
        trace!(self.log, "intersected geometry at: {:?}", isect.general.p);

        isect.compute_scattering_functions(ray, TransportMode::Radiance);

        let n = isect.shading.n;
        let wo = isect.general.wo;

        l += isect.le(&wo);

        trace!(self.log, "L: {:?}, before light rays", l);

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
                        l += f * li * wi.dot(&n).abs() / pdf;
                    } else {
                        trace!(self.log, "light: {:p}, occluded", light);
                    }
                }
            }
        }

        trace!(self.log, "L: {:?}, after light rays", l);

        if depth + 1 < self.max_depth {
            l += self.specular_reflect(&ray, &isect, &scene, &mut sampler, depth);
            l += self.specular_transmit(&ray, &isect, &scene, &mut sampler, depth);
        }

        l
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
        let mut sampler_builder = self.sampler_builder.clone();
        let mut pixel_sampler = sampler_builder.with_seed(0).build();
        pixel_sampler.start_pixel(&pixel);

        loop {
            let camera_sample = pixel_sampler.get_camera_sample(&pixel);
            trace!(self.log, "generated camera sample: {:?}", camera_sample);
            let mut ray = camera.generate_ray_differential(&camera_sample);
            ray.scale_differentials(1.0 / (pixel_sampler.samples_per_pixel() as f32).sqrt());
            trace!(self.log, "generated ray: {:?}", ray);
            let mut l = Spectrum::new(0.0);
            l = self.li(&ray, &scene, &mut pixel_sampler, 0);
            trace!(self.log, "output L: {:?}", l);

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
                let mut tile_sampler = self.sampler_builder.clone().with_seed(seed).build();

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

                        let mut l = Spectrum::new(0.0);
                        l = self.li(&ray, &scene, &mut tile_sampler, 0);

                        film_tile.add_sample(&camera_sample.p_film, &l);

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
