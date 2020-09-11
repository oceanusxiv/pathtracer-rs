use rand::Rng;

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
