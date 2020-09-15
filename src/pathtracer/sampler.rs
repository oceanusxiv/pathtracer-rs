use super::{
    sampling::latin_hyper_cube_2d, sampling::shuffle, sampling::stratified_sample_1d,
    sampling::stratified_sample_2d, sampling::Random, CameraSample,
};
use rand::{Rng, SeedableRng};
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

#[derive(Clone)]
pub struct StratifiedSamplerBuilder {
    x_pixel_samples: usize,
    y_pixel_samples: usize,
    jitter_samples: bool,
    n_sampled_dimensions: usize,
    rng: Random,
    sample_1d_array_sizes: Vec<usize>,
    sample_2d_array_sizes: Vec<usize>,
    sample_array_1d: Vec<Vec<f32>>,
    sample_array_2d: Vec<Vec<na::Point2<f32>>>,
    log: slog::Logger,
}

impl StratifiedSamplerBuilder {
    pub fn new(
        log: &slog::Logger,
        x_pixel_samples: usize,
        y_pixel_samples: usize,
        jitter_samples: bool,
        n_sampled_dimensions: usize,
    ) -> Self {
        let log = log.new(o!("module" => "sampler"));
        Self {
            x_pixel_samples,
            y_pixel_samples,
            jitter_samples,
            n_sampled_dimensions,
            rng: Random::from_entropy(),
            sample_1d_array_sizes: vec![],
            sample_2d_array_sizes: vec![],
            sample_array_1d: vec![],
            sample_array_2d: vec![],
            log,
        }
    }

    pub fn build(&self) -> StratifiedSampler {
        let samples_per_pixel = self.x_pixel_samples * self.y_pixel_samples;
        StratifiedSampler {
            pixel_sampler: PixelSampler::new(
                CoreSampler::new(
                    samples_per_pixel,
                    self.sample_1d_array_sizes.clone(),
                    self.sample_2d_array_sizes.clone(),
                    self.sample_array_1d.clone(),
                    self.sample_array_2d.clone(),
                ),
                samples_per_pixel,
                self.n_sampled_dimensions,
                self.rng.clone(),
            ),
            x_pixel_samples: self.x_pixel_samples,
            y_pixel_samples: self.y_pixel_samples,
            jitter_samples: self.jitter_samples,
            log: self.log.clone(),
        }
    }

    pub fn request_1d_array(&mut self, n: usize) -> &mut Self {
        let samples_per_pixel = self.x_pixel_samples * self.y_pixel_samples;

        self.sample_1d_array_sizes.push(n);
        self.sample_array_1d.push(vec![0.0; n * samples_per_pixel]);

        self
    }

    pub fn request_2d_array(&mut self, n: usize) -> &mut Self {
        let samples_per_pixel = self.x_pixel_samples * self.y_pixel_samples;

        self.sample_2d_array_sizes.push(n);
        self.sample_array_2d
            .push(vec![na::Point2::new(0.0, 0.0); n * samples_per_pixel]);

        self
    }

    pub fn with_seed(&mut self, seed: u64) -> &mut Self {
        self.rng = Random::seed_from_u64(seed);

        self
    }
}

pub struct StratifiedSampler {
    pixel_sampler: PixelSampler,
    x_pixel_samples: usize,
    y_pixel_samples: usize,
    jitter_samples: bool,
    log: slog::Logger,
}

impl StratifiedSampler {
    pub fn start_pixel(&mut self, p: &na::Point2<i32>) {
        let pixel_sampler = &mut self.pixel_sampler;
        let sampler = &mut pixel_sampler.sampler;
        for i in 0..pixel_sampler.samples_1d.len() {
            stratified_sample_1d(
                &mut pixel_sampler.samples_1d[i][..],
                self.x_pixel_samples * self.y_pixel_samples,
                pixel_sampler.rng.get_mut(),
                self.jitter_samples,
            );
            shuffle(
                &mut pixel_sampler.samples_1d[i][..],
                self.x_pixel_samples * self.y_pixel_samples,
                1,
                pixel_sampler.rng.get_mut(),
            );
        }
        for i in 0..pixel_sampler.samples_2d.len() {
            stratified_sample_2d(
                &mut pixel_sampler.samples_2d[i][..],
                self.x_pixel_samples,
                self.y_pixel_samples,
                pixel_sampler.rng.get_mut(),
                self.jitter_samples,
            );
            shuffle(
                &mut pixel_sampler.samples_2d[i][..],
                self.x_pixel_samples * self.y_pixel_samples,
                1,
                pixel_sampler.rng.get_mut(),
            );
        }

        for i in 0..sampler.sample_1d_array_sizes.len() {
            for j in 0..sampler.samples_per_pixel {
                let count = sampler.sample_1d_array_sizes[i];
                stratified_sample_1d(
                    &mut sampler.sample_array_1d[i][j * count..],
                    count,
                    pixel_sampler.rng.get_mut(),
                    self.jitter_samples,
                );
                shuffle(
                    &mut sampler.sample_array_1d[i][j * count..],
                    count,
                    1,
                    pixel_sampler.rng.get_mut(),
                );
            }
        }
        for i in 0..sampler.sample_2d_array_sizes.len() {
            for j in 0..sampler.samples_per_pixel {
                let count = sampler.sample_2d_array_sizes[i];
                latin_hyper_cube_2d(
                    &mut sampler.sample_array_2d[i][j * count..],
                    count,
                    2,
                    pixel_sampler.rng.get_mut(),
                );
            }
        }

        trace!(
            self.log,
            "generated 2d sample array: {:?}",
            pixel_sampler.samples_2d
        );
        sampler.start_pixel(&p);
    }

    pub fn get_camera_sample(&mut self, p_raster: &na::Point2<i32>) -> CameraSample {
        self.pixel_sampler.get_camera_sample(&p_raster)
    }

    pub fn start_next_sample(&mut self) -> bool {
        trace!(self.log, "starting next sample");
        self.pixel_sampler.start_next_sample()
    }

    pub fn get_1d(&self) -> f32 {
        self.pixel_sampler.get_1d()
    }

    pub fn get_2d(&self) -> na::Point2<f32> {
        trace!(
            self.log,
            "curr_2d_dim: {:?}, curr_pixel_sample_idx: {:?}",
            self.pixel_sampler.current_2d_dimension,
            self.pixel_sampler.sampler.current_pixel_sample_index
        );
        let sample = self.pixel_sampler.get_2d();
        trace!(self.log, "2d sample {:?}", sample);
        sample
    }

    pub fn samples_per_pixel(&self) -> usize {
        self.pixel_sampler.sampler.samples_per_pixel
    }

    pub fn get_1d_array(&self, n: usize) -> Option<&[f32]> {
        self.pixel_sampler.sampler.get_1d_array(n)
    }

    pub fn get_2d_array(&self, n: usize) -> Option<&[na::Point2<f32>]> {
        self.pixel_sampler.sampler.get_2d_array(n)
    }

    pub fn get_current_sample_number(&self) -> usize {
        self.pixel_sampler.sampler.current_pixel_sample_index
    }
}

// TODO: add more samplers
pub type Sampler = StratifiedSampler; // for now, since dealing with sampler inheritance is annoying
pub type SamplerBuilder = StratifiedSamplerBuilder;
