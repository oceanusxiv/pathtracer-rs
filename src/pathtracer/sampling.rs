use super::CameraSample;
use rand::prelude::*;

const ONE_MINUS_EPSILON: f32 = hexf32!("0x1.fffffep-1");

#[derive(Clone)]
struct CoreSampler {
    samples_per_pixel: usize,

    current_pixel: na::Point2<i32>,
    current_pixel_sample_index: usize,
    sample_1d_array_sizes: Vec<usize>,
    sample_2d_array_sizes: Vec<usize>,
    sample_array_1d: Vec<Vec<f32>>,
    sample_array_2d: Vec<Vec<na::Point2<f32>>>,

    array_1d_offset: usize,
    array_2d_offset: usize,
}

impl CoreSampler {
    fn new(samples_per_pixel: usize) -> Self {
        CoreSampler {
            samples_per_pixel,
            current_pixel: na::Point2::new(0, 0),
            current_pixel_sample_index: 0,
            sample_1d_array_sizes: vec![],
            sample_2d_array_sizes: vec![],
            sample_array_1d: vec![],
            sample_array_2d: vec![],
            array_1d_offset: 0,
            array_2d_offset: 0,
        }
    }
    fn start_pixel(&mut self, p: &na::Point2<i32>) {
        self.current_pixel = *p;
        self.current_pixel_sample_index = 0;
        self.array_1d_offset = 0;
        self.array_2d_offset = 0;
    }

    fn request_1d_array(&mut self, n: usize) {
        self.sample_1d_array_sizes.push(n);
        self.sample_array_1d
            .push(vec![0.0; n * self.samples_per_pixel])
    }
    fn request_2d_array(&mut self, n: usize) {
        self.sample_2d_array_sizes.push(n);
        self.sample_array_2d
            .push(vec![na::Point2::new(0.0, 0.0); n * self.samples_per_pixel])
    }
    fn round_count(&self, n: i32) -> i32 {
        n
    }
    fn get_1d_array(&mut self, n: usize) -> Option<f32> {
        if self.array_1d_offset == self.sample_array_1d.len() {
            None
        } else {
            let ret =
                self.sample_array_1d[self.array_1d_offset][self.current_pixel_sample_index * n];
            self.array_1d_offset += 1;

            Some(ret)
        }
    }
    fn get_2d_array(&mut self, n: usize) -> Option<na::Point2<f32>> {
        if self.array_2d_offset == self.sample_array_2d.len() {
            None
        } else {
            let ret =
                self.sample_array_2d[self.array_2d_offset][self.current_pixel_sample_index * n];
            self.array_2d_offset += 1;

            Some(ret)
        }
    }
    fn start_next_sample(&mut self) -> bool {
        self.array_1d_offset = 0;
        self.array_2d_offset = 0;
        self.current_pixel_sample_index += 1;
        self.current_pixel_sample_index < self.samples_per_pixel
    }

    fn set_sample_number(&mut self, sample_num: usize) -> bool {
        self.array_1d_offset = 0;
        self.array_2d_offset = 0;
        self.current_pixel_sample_index = sample_num;

        self.current_pixel_sample_index < self.samples_per_pixel
    }
}

#[derive(Clone)]
struct PixelSampler {
    sampler: CoreSampler,
    samples_1d: Vec<Vec<f32>>,
    samples_2d: Vec<Vec<na::Point2<f32>>>,
    current_1d_dimension: usize,
    current_2d_dimension: usize,

    rng: StdRng,
}

impl PixelSampler {
    fn new(samples_per_pixel: usize, n_sampled_dimensions: usize) -> Self {
        let samples_1d = vec![vec![0.0; samples_per_pixel]; n_sampled_dimensions];
        let samples_2d: Vec<Vec<na::Point2<f32>>> =
            vec![vec![na::Point2::new(0.0, 0.0); samples_per_pixel]; n_sampled_dimensions];

        PixelSampler {
            sampler: CoreSampler::new(samples_per_pixel),
            samples_1d,
            samples_2d,
            current_1d_dimension: 0,
            current_2d_dimension: 0,
            rng: rand::rngs::StdRng::from_entropy(),
        }
    }

    fn start_next_sample(&mut self) -> bool {
        self.current_1d_dimension = 0;
        self.current_2d_dimension = 0;
        self.sampler.start_next_sample()
    }

    fn set_sample_number(&mut self, sample_num: usize) -> bool {
        self.current_1d_dimension = 0;
        self.current_2d_dimension = 0;
        self.sampler.set_sample_number(sample_num)
    }

    fn get_1d(&mut self) -> f32 {
        if self.current_1d_dimension < self.samples_1d.len() {
            let ret =
                self.samples_1d[self.current_1d_dimension][self.sampler.current_pixel_sample_index];
            self.current_1d_dimension += 1;
            ret
        } else {
            self.rng.gen_range(0.0, 1.0)
        }
    }

    fn get_2d(&mut self) -> na::Point2<f32> {
        if self.current_2d_dimension < self.samples_2d.len() {
            let ret =
                self.samples_2d[self.current_2d_dimension][self.sampler.current_pixel_sample_index];
            self.current_2d_dimension += 1;
            ret
        } else {
            na::Point2::new(self.rng.gen_range(0.0, 1.0), self.rng.gen_range(0.0, 1.0))
        }
    }

    fn get_camera_sample(&mut self, p_raster: &na::Point2<i32>) -> CameraSample {
        CameraSample {
            p_film: na::Point2::new(p_raster.x as f32, p_raster.y as f32) + self.get_2d().coords,
        }
    }
}

fn stratified_sample_1d(samp: &mut [f32], n_samples: usize, rng: &mut StdRng, jitter: bool) {
    let inv_n_samples = 1.0 / (n_samples as f32);

    for i in 0..n_samples {
        let delta = if jitter {
            rng.gen_range(0.0, 1.0)
        } else {
            0.5f32
        };
        samp[i] = ((i as f32 + delta) * inv_n_samples).min(ONE_MINUS_EPSILON);
    }
}

fn stratified_sample_2d(
    samp: &mut [na::Point2<f32>],
    nx: usize,
    ny: usize,
    rng: &mut StdRng,
    jitter: bool,
) {
    let dx = 1.0 / (nx as f32);
    let dy = 1.0 / (ny as f32);

    let mut i = 0;
    for y in 0..ny {
        for x in 0..nx {
            let jx = if jitter {
                rng.gen_range(0.0, 1.0)
            } else {
                0.5f32
            };
            let jy = if jitter {
                rng.gen_range(0.0, 1.0)
            } else {
                0.5f32
            };
            samp[i].x = ((i as f32 + jx) * dx).min(ONE_MINUS_EPSILON);
            samp[i].y = ((i as f32 + jy) * dy).min(ONE_MINUS_EPSILON);
            i += 1;
        }
    }
}

fn shuffle<T>(samp: &mut [T], count: usize, n_dimensions: usize, rng: &mut StdRng) {
    for i in 0..count {
        let other = i + rng.gen_range(0, count - i);

        for j in 0..n_dimensions {
            samp.swap(n_dimensions * i + j, n_dimensions * other + j);
        }
    }
}

fn latin_hyper_cube_2d(
    samples: &mut [na::Point2<f32>],
    n_samples: usize,
    n_dim: usize,
    rng: &mut StdRng,
) {
    let inv_n_samples = 1.0 / (n_samples as f32);
    for i in 0..n_samples {
        for j in 0..n_dim {
            let sj = (i as f32 + rng.gen::<f32>()) * inv_n_samples;
            samples[i][j] = sj.min(ONE_MINUS_EPSILON);
        }
    }

    for i in 0..n_dim {
        for j in 0..n_samples {
            let other = j + rng.gen_range(0, n_samples - j);
            let tmp = samples[j][i];
            samples[j][i] = samples[other][i];
            samples[other][i] = tmp;
        }
    }
}

#[derive(Clone)]
pub struct StratifiedSampler {
    pixel_sampler: PixelSampler,
    x_pixel_samples: usize,
    y_pixel_samples: usize,
    jitter_samples: bool,
}

impl StratifiedSampler {
    pub fn new(
        x_pixel_samples: usize,
        y_pixel_samples: usize,
        jitter_samples: bool,
        n_sampled_dimensions: usize,
    ) -> Self {
        StratifiedSampler {
            pixel_sampler: PixelSampler::new(
                x_pixel_samples * y_pixel_samples,
                n_sampled_dimensions,
            ),
            x_pixel_samples,
            y_pixel_samples,
            jitter_samples,
        }
    }

    pub fn start_pixel(&mut self, p: &na::Point2<i32>) {
        let pixel_sampler = &mut self.pixel_sampler;
        let sampler = &mut pixel_sampler.sampler;
        for i in 0..pixel_sampler.samples_1d.len() {
            stratified_sample_1d(
                &mut pixel_sampler.samples_1d[i][..],
                self.x_pixel_samples * self.y_pixel_samples,
                &mut pixel_sampler.rng,
                self.jitter_samples,
            );
            shuffle(
                &mut pixel_sampler.samples_1d[i][..],
                self.x_pixel_samples * self.y_pixel_samples,
                1,
                &mut pixel_sampler.rng,
            );
        }
        for i in 0..pixel_sampler.samples_2d.len() {
            stratified_sample_2d(
                &mut pixel_sampler.samples_2d[i][..],
                self.x_pixel_samples,
                self.y_pixel_samples,
                &mut pixel_sampler.rng,
                self.jitter_samples,
            );
            shuffle(
                &mut pixel_sampler.samples_2d[i][..],
                self.x_pixel_samples * self.y_pixel_samples,
                1,
                &mut pixel_sampler.rng,
            );
        }

        for i in 0..sampler.sample_1d_array_sizes.len() {
            for j in 0..sampler.samples_per_pixel {
                let count = sampler.sample_1d_array_sizes[i];
                stratified_sample_1d(
                    &mut sampler.sample_array_1d[i][j * count..],
                    count,
                    &mut pixel_sampler.rng,
                    self.jitter_samples,
                );
                shuffle(
                    &mut sampler.sample_array_1d[i][j * count..],
                    count,
                    1,
                    &mut pixel_sampler.rng,
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
                    &mut pixel_sampler.rng,
                );
            }
        }

        sampler.start_pixel(&p);
    }

    pub fn clone_with_seed(&self, seed: u64) -> Self {
        let mut ret = self.clone();
        ret.pixel_sampler.rng = rand::rngs::StdRng::seed_from_u64(seed);

        ret
    }

    pub fn get_camera_sample(&mut self, p_raster: &na::Point2<i32>) -> CameraSample {
        self.pixel_sampler.get_camera_sample(&p_raster)
    }

    pub fn start_next_sample(&mut self) -> bool {
        self.pixel_sampler.sampler.start_next_sample()
    }

    pub fn get_2d(&mut self) -> na::Point2<f32> {
        self.pixel_sampler.get_2d()
    }
}

pub type Sampler = StratifiedSampler; // for now, since dealing with sampler inheritance is annoying

fn uniform_sample_hemisphere(u: &glm::Vec2) -> glm::Vec3 {
    glm::vec3(0.0, 0.0, 0.0)
}

pub fn concentric_sample_disk(u: &na::Point2<f32>) -> na::Point2<f32> {
    let u_offset = 2.0 * u - na::Vector2::new(1.0, 1.0);

    if u_offset.x == 0.0 && u_offset.y == 0.0 {
        na::Point2::new(0.0, 0.0)
    } else {
        let mut theta = 0.0f32;
        let mut r = 0.0f32;

        if u_offset.x.abs() > u_offset.y.abs() {
            r = u_offset.x;
            theta = std::f32::consts::FRAC_PI_4 * (u_offset.y / u_offset.x);
        } else {
            r = u_offset.y;
            theta = std::f32::consts::FRAC_PI_2
                - std::f32::consts::FRAC_PI_4 * (u_offset.x / u_offset.y);
        }

        r * na::Point2::new(theta.cos(), theta.sin())
    }
}

pub fn cosine_sample_hemisphere(u: &na::Point2<f32>) -> na::Vector3<f32> {
    let d = concentric_sample_disk(&u);
    let z = 0.0f32.max(1.0 - d.x * d.x - d.y * d.y).sqrt();
    na::Vector3::new(d.x, d.y, z)
}

pub fn cosine_hemisphere_pdf(cos_theta: f32) -> f32 {
    cos_theta * std::f32::consts::FRAC_1_PI
}
