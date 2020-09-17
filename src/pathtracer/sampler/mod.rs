pub mod sobol;
pub mod stratified;

use super::{sampling::Random, CameraSample};
use rand::Rng;
use std::cell::{Cell, RefCell};

#[derive(Clone)]
struct CoreSampler {
    samples_per_pixel: usize,

    current_pixel: na::Point2<i32>,
    current_pixel_sample_index: usize,
    sample_1d_array_sizes: Vec<usize>,
    sample_2d_array_sizes: Vec<usize>,
    sample_array_1d: Vec<Vec<f32>>,
    sample_array_2d: Vec<Vec<na::Point2<f32>>>,
    array_1d_offset: Cell<usize>,
    array_2d_offset: Cell<usize>,
}

impl CoreSampler {
    fn new(
        samples_per_pixel: usize,
        sample_1d_array_sizes: Vec<usize>,
        sample_2d_array_sizes: Vec<usize>,
        sample_array_1d: Vec<Vec<f32>>,
        sample_array_2d: Vec<Vec<na::Point2<f32>>>,
    ) -> Self {
        Self {
            samples_per_pixel,
            current_pixel: na::Point2::new(0, 0),
            current_pixel_sample_index: 0,
            sample_1d_array_sizes,
            sample_2d_array_sizes,
            sample_array_1d,
            sample_array_2d,
            array_1d_offset: Cell::new(0),
            array_2d_offset: Cell::new(0),
        }
    }
    fn start_pixel(&mut self, p: &na::Point2<i32>) {
        self.current_pixel = *p;
        self.current_pixel_sample_index = 0;
        self.array_1d_offset.set(0);
        self.array_2d_offset.set(0)
    }

    fn round_count(&self, n: i32) -> i32 {
        n
    }

    fn get_1d_array(&self, n: usize) -> Option<&[f32]> {
        if self.array_1d_offset.get() == self.sample_array_1d.len() {
            None
        } else {
            let ret = &self.sample_array_1d[self.array_1d_offset.get()]
                [self.current_pixel_sample_index * n..];
            self.array_1d_offset.set(self.array_1d_offset.get() + 1);

            Some(ret)
        }
    }

    fn get_2d_array(&self, n: usize) -> Option<&[na::Point2<f32>]> {
        if self.array_2d_offset.get() == self.sample_array_2d.len() {
            None
        } else {
            let ret = &self.sample_array_2d[self.array_2d_offset.get()]
                [self.current_pixel_sample_index * n..];
            self.array_2d_offset.set(self.array_2d_offset.get() + 1);

            Some(ret)
        }
    }

    fn start_next_sample(&mut self) -> bool {
        self.array_1d_offset.set(0);
        self.array_2d_offset.set(0);
        self.current_pixel_sample_index += 1;
        self.current_pixel_sample_index < self.samples_per_pixel
    }

    fn set_sample_number(&mut self, sample_num: usize) -> bool {
        self.array_1d_offset.set(0);
        self.array_2d_offset.set(0);
        self.current_pixel_sample_index = sample_num;

        self.current_pixel_sample_index < self.samples_per_pixel
    }
}

#[derive(Clone)]
struct PixelSampler {
    sampler: CoreSampler,
    samples_1d: Vec<Vec<f32>>,
    samples_2d: Vec<Vec<na::Point2<f32>>>,
    current_1d_dimension: Cell<usize>,
    current_2d_dimension: Cell<usize>,

    rng: RefCell<Random>,
}

impl PixelSampler {
    fn new(
        sampler: CoreSampler,
        samples_per_pixel: usize,
        n_sampled_dimensions: usize,
        rng: Random,
    ) -> Self {
        let samples_1d = vec![vec![0.0; samples_per_pixel]; n_sampled_dimensions];
        let samples_2d: Vec<Vec<na::Point2<f32>>> =
            vec![vec![na::Point2::new(0.0, 0.0); samples_per_pixel]; n_sampled_dimensions];

        Self {
            sampler,
            samples_1d,
            samples_2d,
            current_1d_dimension: Cell::new(0),
            current_2d_dimension: Cell::new(0),
            rng: RefCell::new(rng),
        }
    }

    fn start_next_sample(&mut self) -> bool {
        self.current_1d_dimension.set(0);
        self.current_2d_dimension.set(0);
        self.sampler.start_next_sample()
    }

    fn set_sample_number(&mut self, sample_num: usize) -> bool {
        self.current_1d_dimension.set(0);
        self.current_2d_dimension.set(0);
        self.sampler.set_sample_number(sample_num)
    }

    fn get_1d(&self) -> f32 {
        if self.current_1d_dimension.get() < self.samples_1d.len() {
            let ret = self.samples_1d[self.current_1d_dimension.get()]
                [self.sampler.current_pixel_sample_index];
            self.current_1d_dimension
                .set(self.current_1d_dimension.get() + 1);
            ret
        } else {
            self.rng.borrow_mut().gen_range(0.0, 1.0)
        }
    }

    fn get_2d(&self) -> na::Point2<f32> {
        if self.current_2d_dimension.get() < self.samples_2d.len() {
            let ret = self.samples_2d[self.current_2d_dimension.get()]
                [self.sampler.current_pixel_sample_index];
            self.current_2d_dimension
                .set(self.current_2d_dimension.get() + 1);
            ret
        } else {
            let mut rng = self.rng.borrow_mut();
            na::Point2::new(rng.gen_range(0.0, 1.0), rng.gen_range(0.0, 1.0))
        }
    }

    fn get_camera_sample(&mut self, p_raster: &na::Point2<i32>) -> CameraSample {
        CameraSample {
            p_film: na::Point2::new(p_raster.x as f32, p_raster.y as f32) + self.get_2d().coords,
        }
    }
}

// TODO: add more samplers
pub type Sampler = stratified::StratifiedSampler; // for now, since dealing with sampler inheritance is annoying
pub type SamplerBuilder = stratified::StratifiedSamplerBuilder;
