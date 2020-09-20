use std::cell::Cell;

use crate::{common::bounds::Bounds2i, pathtracer::CameraSample};

use super::CoreSampler;
use crate::common::math::{
    cantor_pairing, log2_int, RoundUpPow2, HALF_MAX_I_32, ONE_MINUS_EPSILON,
};
use crate::pathtracer::lowdiscrepancy::{sobol_interval_to_index, sobol_sample};
use crate::pathtracer::sobolmatrices::NUM_SOBOL_DIMENSIONS;
const ARRAY_START_DIM: usize = 5;

pub struct SobolSampler {
    sampler: CoreSampler,
    dimension: Cell<usize>,
    interval_sample_index: i64,
    array_end_dim: usize,

    sample_bounds: Bounds2i,
    resolution: i32,
    log_2_resolution: u32,
    current_scramble_index: u64,
}

#[derive(Clone)]
pub struct SobolSamplerBuilder {
    samples_per_pixel: usize,
    sample_bounds: Bounds2i,
    resolution: i32,
    log_2_resolution: u32,
    log: slog::Logger,
}

impl SobolSamplerBuilder {
    pub fn new(log: &slog::Logger, samples_per_pixel: usize, sample_bounds: &Bounds2i) -> Self {
        let log = log.new(o!("module" => "sampler"));
        let new_samples_per_pixel = (samples_per_pixel as i64).round_up_pow_2();
        let diag = sample_bounds.diagonal();
        let resolution = diag.x.max(diag.y).round_up_pow_2();
        let log_2_resolution = log2_int(resolution as usize);

        if resolution > 0 {
            debug_assert_eq!(1 << log_2_resolution, resolution);
        }

        if !samples_per_pixel.is_power_of_two() {
            warn!(
                log,
                "non power-of-two sample count rounded up to {:?} for sobol sampler",
                new_samples_per_pixel
            );
        }
        Self {
            samples_per_pixel: new_samples_per_pixel as usize,
            sample_bounds: *sample_bounds,
            resolution,
            log_2_resolution,
            log,
        }
    }

    pub fn build(&self) -> SobolSampler {
        SobolSampler {
            sampler: CoreSampler::new(self.samples_per_pixel, vec![], vec![], vec![], vec![]),
            dimension: Cell::new(0),
            interval_sample_index: 0,
            array_end_dim: 0,
            resolution: self.resolution,
            log_2_resolution: self.log_2_resolution,
            sample_bounds: self.sample_bounds,
            current_scramble_index: 0,
        }
    }

    pub fn with_seed(&mut self, _seed: u64) -> &mut Self {
        self
    }
}

impl SobolSampler {
    pub fn start_pixel(&mut self, p: &na::Point2<i32>) {
        self.sampler.start_pixel(p);
        self.current_scramble_index = cantor_pairing(
            (self.sampler.current_pixel.x + HALF_MAX_I_32) as usize,
            (self.sampler.current_pixel.y + HALF_MAX_I_32) as usize,
        ) as u64;
        self.dimension = Cell::new(0);
        self.interval_sample_index = self.get_index_for_sample(0);
        self.array_end_dim = ARRAY_START_DIM
            + self.sampler.sample_array_1d.len()
            + 2 * self.sampler.sample_array_2d.len();

        for i in 0..self.sampler.sample_1d_array_sizes.len() {
            let n_samples = self.sampler.sample_1d_array_sizes[i] * self.sampler.samples_per_pixel;
            for j in 0..n_samples {
                let index = self.get_index_for_sample(j as u64);
                self.sampler.sample_array_1d[i][j] =
                    self.sample_dimension(index, ARRAY_START_DIM + 1);
            }
        }

        let mut dim = ARRAY_START_DIM + self.sampler.sample_1d_array_sizes.len();
        for i in 0..self.sampler.sample_2d_array_sizes.len() {
            let n_samples = self.sampler.sample_2d_array_sizes[i] * self.sampler.samples_per_pixel;
            for j in 0..n_samples {
                let index = self.get_index_for_sample(j as u64);
                self.sampler.sample_array_2d[i][j].x = self.sample_dimension(index, dim);
                self.sampler.sample_array_2d[i][j].x = self.sample_dimension(index, dim + 1);
            }
            dim += 2;
        }

        debug_assert_eq!(self.array_end_dim, dim);
    }

    pub fn get_camera_sample(&mut self, p_raster: &na::Point2<i32>) -> CameraSample {
        CameraSample {
            p_film: na::Point2::new(p_raster.x as f32, p_raster.y as f32) + self.get_2d().coords,
        }
    }

    pub fn start_next_sample(&mut self) -> bool {
        self.dimension.set(0);
        self.interval_sample_index =
            self.get_index_for_sample((self.sampler.current_pixel_sample_index + 1) as u64);
        self.sampler.start_next_sample()
    }

    pub fn get_1d(&self) -> f32 {
        if self.dimension.get() >= ARRAY_START_DIM && self.dimension.get() < self.array_end_dim {
            self.dimension.set(self.array_end_dim);
        }

        let sample = self.sample_dimension(self.interval_sample_index, self.dimension.get());
        self.dimension.set(self.dimension.get() + 1);
        sample
    }

    pub fn get_2d(&self) -> na::Point2<f32> {
        if self.dimension.get() + 1 >= ARRAY_START_DIM && self.dimension.get() < self.array_end_dim
        {
            self.dimension.set(self.array_end_dim);
        }

        let sample = na::Point2::new(
            self.sample_dimension(self.interval_sample_index, self.dimension.get()),
            self.sample_dimension(self.interval_sample_index, self.dimension.get() + 1),
        );
        self.dimension.set(self.dimension.get() + 2);
        sample
    }

    pub fn samples_per_pixel(&self) -> usize {
        self.sampler.samples_per_pixel
    }

    pub fn get_1d_array(&self, n: usize) -> Option<&[f32]> {
        self.sampler.get_1d_array(n)
    }

    pub fn get_2d_array(&self, n: usize) -> Option<&[na::Point2<f32>]> {
        self.sampler.get_2d_array(n)
    }

    pub fn get_current_sample_number(&self) -> usize {
        self.sampler.current_pixel_sample_index
    }

    pub fn get_index_for_sample(&self, sample_num: u64) -> i64 {
        sobol_interval_to_index(
            self.log_2_resolution,
            sample_num,
            &na::Point2::from(self.sampler.current_pixel - self.sample_bounds.p_min),
        ) as i64
    }

    pub fn sample_dimension(&self, index: i64, dimension: usize) -> f32 {
        if dimension > NUM_SOBOL_DIMENSIONS as usize {
            panic!(
                "sobol sampler can only sample up to {:?} dimensions.",
                NUM_SOBOL_DIMENSIONS
            );
        }

        let mut s = sobol_sample(index, dimension, self.current_scramble_index);

        if dimension == 0 || dimension == 1 {
            s = s * self.resolution as f32 + self.sample_bounds.p_min[dimension] as f32;
            s = (s - self.sampler.current_pixel[dimension] as f32).clamp(0., ONE_MINUS_EPSILON);
        }

        s
    }
}
