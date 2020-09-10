use super::interaction::SurfaceMediumInteraction;
use super::sampling::{Sampler, SamplerBuilder};
use super::{bxdf::BxDFType, light::is_delta_light};
use super::{light::SyncLight, RenderScene, TransportMode};
use crate::common::film::FilmTile;
use crate::common::ray::RayDifferential;
use crate::common::spectrum::Spectrum;
use crate::common::Camera;
use crate::common::{bounds::Bounds2i, math::power_heuristic};
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
    handle_media: bool,
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
    let mut scattering_pdf = 0.0;
    let mut visibility = None;
    let mut li = light.sample_li(
        &it.general,
        &u_light,
        &mut wi,
        &mut light_pdf,
        &mut visibility,
    );
    let visibility = visibility.unwrap();
    if light_pdf > 0.0 && !li.is_black() {
        let f: Spectrum;
        if it.is_surface_interaction() {
            let bsdf = it.bsdf.as_ref().unwrap();
            f = bsdf.f(&it.general.wo, &wi, bsdf_flags) * wi.dot(&it.shading.n).abs();
            scattering_pdf = bsdf.pdf(&it.general.wo, &wi, bsdf_flags);
        } else {
            panic!("medium interaction not supported!");
        }

        if !f.is_black() {
            if handle_media {
                panic!("media not supported");
            } else {
                if !visibility.unoccluded(&scene) {
                    li = Spectrum::new(0.0);
                }
            }

            if !li.is_black() {
                if is_delta_light(&light.flags()) {
                    ld += f * li / light_pdf;
                } else {
                    let weight = power_heuristic(1, light_pdf, 1, scattering_pdf);
                    ld += f * li * weight / light_pdf;
                }
            }
        }
    }

    if !is_delta_light(&light.flags()) {
        let mut f;
        let mut sampled_specular = false;

        if it.is_surface_interaction() {
            let mut sampled_type = Some(BxDFType::BSDF_ALL);
            let bsdf = it.bsdf.as_ref().unwrap();
            f = bsdf.sample_f(
                &it.general.wo,
                &mut wi,
                u_scattering,
                &mut scattering_pdf,
                bsdf_flags,
                &mut sampled_type,
            );
            f *= wi.dot(&it.shading.n).abs();
            sampled_specular = sampled_type.unwrap().contains(BxDFType::BSDF_SPECULAR);
        } else {
            panic!("medium interaction not supported!");
        }

        if !f.is_black() && scattering_pdf > 0.0 {
            let mut weight = 1.0;
            if !sampled_specular {
                light_pdf = light.pdf_li(&it.general, &wi);
                if light_pdf == 0.0 {
                    return ld;
                }
                weight = power_heuristic(1, scattering_pdf, 1, light_pdf);
            }

            let mut light_isect = SurfaceMediumInteraction::default();
            let ray = it.general.spawn_ray(&wi);
            let tr = Spectrum::new(1.0);
            let found_surface_interaction = if handle_media {
                panic!("medium interaction not supported!")
            } else {
                scene.intersect(&ray, &mut light_isect)
            };

            let mut li = Spectrum::new(0.0);
            if found_surface_interaction {
                if let Some(isect_light) = light_isect.primitive.unwrap().get_area_light() {
                    if std::ptr::eq(light, isect_light) {
                        li = light_isect.le(&-wi);
                    }
                }
            } else {
                li = light.le(&RayDifferential::new(ray));
            }
            if !li.is_black() {
                ld += f * li * tr * weight / scattering_pdf;
            }
        }
    }

    ld
}

fn uniform_sample_all_lights(
    it: &SurfaceMediumInteraction,
    scene: &RenderScene,
    sampler: &mut Sampler,
    num_light_samples: &Vec<usize>,
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
            l += estimate_direct(
                &it,
                &u_scattering,
                light,
                &u_light,
                &scene,
                &sampler,
                false,
                false,
            );
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
        * estimate_direct(
            &it,
            &u_scattering,
            light,
            &u_light,
            &scene,
            &sampler,
            false,
            false,
        )
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
        if isect.bsdf.is_none() {
            return self.li(
                &RayDifferential::new(isect.general.spawn_ray(&ray.ray.d)),
                &scene,
                &mut sampler,
                depth,
            );
        }

        let wo = isect.general.wo;

        l += isect.le(&wo);

        trace!(self.log, "L: {:?}, before light rays", l);

        if !scene.lights.is_empty() {
            if self.strategy == LightStrategy::UniformSampleAll {
                l += uniform_sample_all_lights(
                    &isect,
                    &scene,
                    &mut sampler,
                    &self.num_light_samples,
                );
            } else {
                l += uniform_sample_one_light(&isect, &scene, &mut sampler);
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
