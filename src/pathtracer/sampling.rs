use rand::Rng;

use crate::common::math::find_interval;

pub const ONE_MINUS_EPSILON: f32 = hexf32!("0x1.fffffep-1");
pub type Random = rand::rngs::SmallRng;

pub fn stratified_sample_1d(samp: &mut [f32], n_samples: usize, rng: &mut Random, jitter: bool) {
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

pub fn stratified_sample_2d(
    samp: &mut [na::Point2<f32>],
    nx: usize,
    ny: usize,
    rng: &mut Random,
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
            samp[i].x = ((x as f32 + jx) * dx).min(ONE_MINUS_EPSILON);
            samp[i].y = ((y as f32 + jy) * dy).min(ONE_MINUS_EPSILON);
            i += 1;
        }
    }
}

pub fn shuffle<T>(samp: &mut [T], count: usize, n_dimensions: usize, rng: &mut Random) {
    for i in 0..count {
        let other = i + rng.gen_range(0, count - i);

        for j in 0..n_dimensions {
            samp.swap(n_dimensions * i + j, n_dimensions * other + j);
        }
    }
}

pub fn latin_hyper_cube_2d(
    samples: &mut [na::Point2<f32>],
    n_samples: usize,
    n_dim: usize,
    rng: &mut Random,
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

pub fn uniform_sample_hemisphere(u: &na::Point2<f32>) -> na::Vector3<f32> {
    let z = u[0];
    let r = 0.0f32.max(1.0 - z * z).sqrt();
    let phi = 2.0 * std::f32::consts::PI * u[1];
    na::Vector3::new(r * phi.cos(), r * phi.sin(), z)
}

pub const fn uniform_hemisphere_pdf() -> f32 {
    const INV_2_PI: f32 = 0.15915494309189533577;
    INV_2_PI
}

pub fn concentric_sample_disk(u: &na::Point2<f32>) -> na::Point2<f32> {
    let u_offset = 2.0 * u - na::Vector2::new(1.0, 1.0);

    if u_offset.x == 0.0 && u_offset.y == 0.0 {
        na::Point2::new(0.0, 0.0)
    } else {
        let theta;
        let r;

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

pub struct Distribution1D {
    func: Vec<f32>,
    cdf: Vec<f32>,
    func_int: f32,
}

impl Distribution1D {
    pub fn new(f: &[f32], n: usize) -> Self {
        let mut cdf = vec![0.0; n + 1];
        for i in 1..(n + 1) {
            cdf[i] = cdf[i - 1] + f[i - 1] / n as f32;
        }

        let func_int = cdf[n];

        if func_int == 0.0 {
            for i in 1..(n + 1) {
                cdf[i] = i as f32 / n as f32;
            }
        } else {
            for i in 1..(n + 1) {
                cdf[i] /= func_int;
            }
        }

        Self {
            func: f.to_vec(),
            cdf,
            func_int,
        }
    }

    fn count(&self) -> usize {
        self.func.len()
    }

    pub fn sample_continuous(&self, u: f32, pdf: &mut f32, off: &mut Option<usize>) -> f32 {
        let offset = find_interval(self.cdf.len(), |index| self.cdf[index] <= u);
        if let Some(off) = off {
            *off = offset;
        }

        let mut du = u - self.cdf[offset];
        if (self.cdf[offset + 1] - self.cdf[offset]) > 0.0 {
            du /= self.cdf[offset + 1] - self.cdf[offset];
        }

        *pdf = if self.func_int > 0.0 {
            self.func[offset] / self.func_int
        } else {
            0.0
        };

        (offset as f32 + du) / (self.count() as f32)
    }
}

pub struct Distribution2D {
    p_conditional_v: Vec<Box<Distribution1D>>,
    p_marginal: Box<Distribution1D>,
}

impl Distribution2D {
    pub fn new(func: &[f32], nu: usize, nv: usize) -> Self {
        let mut p_conditional_v = Vec::with_capacity(nv);
        for v in 0..nv {
            p_conditional_v.push(Box::new(Distribution1D::new(
                &func[v * nu..(nu * (v + 1))],
                nu,
            )));
        }

        let mut marginal_func = Vec::with_capacity(nv);
        for v in 0..nv {
            marginal_func.push(p_conditional_v[v].func_int);
        }

        Self {
            p_conditional_v,
            p_marginal: Box::new(Distribution1D::new(&marginal_func[..], nv)),
        }
    }

    pub fn sample_continuous(&self, u: &na::Point2<f32>, pdf: &mut f32) -> na::Point2<f32> {
        let mut pdfs = [0.0, 0.0];
        let mut v = Some(1);
        let d1 = self
            .p_marginal
            .sample_continuous(u[1], &mut pdfs[1], &mut v);
        let v = v.unwrap();
        let d0 = self.p_conditional_v[v].sample_continuous(u[0], &mut pdfs[0], &mut None);
        *pdf = pdfs[0] * pdfs[1];
        na::Point2::new(d0, d1)
    }

    pub fn pdf(&self, p: &na::Point2<f32>) -> f32 {
        let iu = ((p[0] * self.p_conditional_v[0].count() as f32) as usize)
            .clamp(0, self.p_conditional_v[0].count() - 1);
        let iv = ((p[1] * self.p_marginal.count() as f32) as usize)
            .clamp(0, self.p_marginal.count() - 1);
        self.p_conditional_v[iv].func[iu] / self.p_marginal.func_int
    }
}
