use crate::pathtracer::CameraSample;

use super::CoreSampler;

const ARRAY_START_DIM: usize = 5;

pub struct SobolSampler {
    sampler: CoreSampler,
    dimension: usize,
    interval_sample_index: usize,
    array_end_dim: usize,
}

impl SobolSampler {
    pub fn start_pixel(&mut self, p: &na::Point2<i32>) {
        todo!()
    }

    pub fn get_camera_sample(&mut self, p_raster: &na::Point2<i32>) -> CameraSample {
        todo!()
    }

    pub fn start_next_sample(&mut self) -> bool {
        todo!()
    }

    pub fn get_1d(&self) -> f32 {
        todo!()
    }

    pub fn get_2d(&self) -> na::Point2<f32> {
        todo!()
    }

    pub fn samples_per_pixel(&self) -> usize {
        todo!()
    }

    pub fn get_1d_array(&self, n: usize) -> Option<&[f32]> {
        todo!()
    }

    pub fn get_2d_array(&self, n: usize) -> Option<&[na::Point2<f32>]> {
        todo!()
    }

    pub fn get_current_sample_number(&self) -> usize {
        todo!()
    }

    pub fn get_index_for_sample(&self, sample_num: usize) -> usize {
        todo!()
    }

    pub fn sample_dimension(&self, index: usize, dimension: usize) -> f32 {
        todo!()
    }
}
