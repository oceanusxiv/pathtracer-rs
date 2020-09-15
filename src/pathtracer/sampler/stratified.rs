use super::{CoreSampler, PixelSampler};
use crate::pathtracer::{
    sampling::latin_hyper_cube_2d, sampling::shuffle, sampling::stratified_sample_1d,
    sampling::stratified_sample_2d, sampling::Random, CameraSample,
};
use rand::SeedableRng;

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
