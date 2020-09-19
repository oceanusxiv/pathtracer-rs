use std::cell::Cell;

use crate::pathtracer::CameraSample;

use super::CoreSampler;

const ARRAY_START_DIM: usize = 5;

pub struct SobolSampler {
    sampler: CoreSampler,
    dimension: Cell<usize>,
    interval_sample_index: u64,
    array_end_dim: usize,
}

impl SobolSampler {
    pub fn start_pixel(&mut self, p: &na::Point2<i32>) {
        self.sampler.start_pixel(p);
        self.dimension = Cell::new(0);
        self.interval_sample_index = self.get_index_for_sample(0);
        self.array_end_dim = ARRAY_START_DIM
            + self.sampler.sample_array_1d.len()
            + 2 * self.sampler.sample_array_2d.len();

        for i in 0..self.sampler.sample_1d_array_sizes.len() {
            let n_samples = self.sampler.sample_1d_array_sizes[i] * self.sampler.samples_per_pixel;
            for j in 0..n_samples {
                let index = self.get_index_for_sample(j);
                self.sampler.sample_array_1d[i][j] =
                    self.sample_dimension(index, ARRAY_START_DIM + 1);
            }
        }

        let mut dim = ARRAY_START_DIM + self.sampler.sample_1d_array_sizes.len();
        for i in 0..self.sampler.sample_2d_array_sizes.len() {
            let n_samples = self.sampler.sample_2d_array_sizes[i] * self.sampler.samples_per_pixel;
            for j in 0..n_samples {
                let index = self.get_index_for_sample(j);
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
            self.get_index_for_sample(self.sampler.current_pixel_sample_index + 1);
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

    pub fn get_index_for_sample(&self, sample_num: usize) -> u64 {
        todo!()
    }

    pub fn sample_dimension(&self, index: u64, dimension: usize) -> f32 {
        todo!()
    }
}
