use std::ops::{AddAssign, Mul};

use super::interaction::SurfaceMediumInteraction;
use crate::common::{
    math::abs_mod, math::lerp, math::log2_int, math::round_up_pow_2, spectrum::Spectrum, WrapMode,
};

pub trait Texture<T> {
    fn evaluate(&self, it: &SurfaceMediumInteraction) -> T;
}

pub trait SyncTexture<T>: Texture<T> + Send + Sync {}
impl<T1, T2> SyncTexture<T1> for T2 where T2: Texture<T1> + Send + Sync {}

pub struct ConstantTexture<T: Copy> {
    value: T,
}

impl<T: Copy> ConstantTexture<T> {
    pub fn new(value: T) -> Self {
        Self { value }
    }
}

impl<T: Copy> Texture<T> for ConstantTexture<T> {
    fn evaluate(&self, _it: &SurfaceMediumInteraction) -> T {
        self.value
    }
}

pub struct UVMap {
    su: f32,
    sv: f32,
    du: f32,
    dv: f32,
}

impl UVMap {
    pub fn new(su: f32, sv: f32, du: f32, dv: f32) -> Self {
        Self { su, sv, du, dv }
    }

    pub fn map(
        &self,
        it: &SurfaceMediumInteraction,
        dst_dx: &mut na::Vector2<f32>,
        dst_dy: &mut na::Vector2<f32>,
    ) -> na::Point2<f32> {
        *dst_dx = na::Vector2::new(self.su * it.dudx, self.sv * it.dvdx);
        *dst_dy = na::Vector2::new(self.su * it.dudy, self.sv * it.dvdy);

        na::Point2::new(self.su * it.uv[0] + self.du, self.sv * it.uv[1] + self.dv)
    }
}

pub struct ImageTexture<T: na::Scalar + num::Zero> {
    mip_map: MIPMap<T>,
    mapping: UVMap,
    log: slog::Logger,
}

impl ImageTexture<f32> {
    pub fn new(
        log: &slog::Logger,
        image: &image::GrayImage,
        scale: f32,
        wrap_mode: WrapMode,
        mapping: UVMap,
    ) -> Self {
        let matrix = na::DMatrix::from_fn(
            image.height() as usize,
            image.width() as usize,
            |row, col| scale * (image.get_pixel(col as u32, row as u32)[0] as f32 / 255.0),
        );

        let log = log.new(o!());

        Self {
            mip_map: MIPMap::new(&log, matrix, true, wrap_mode),
            mapping,
            log,
        }
    }
}

impl ImageTexture<Spectrum> {
    pub fn new(
        log: &slog::Logger,
        image: &image::RgbImage,
        scale: Spectrum,
        wrap_mode: WrapMode,
        mapping: UVMap,
        gamma: bool,
    ) -> Self {
        let matrix = na::DMatrix::from_fn(
            image.height() as usize,
            image.width() as usize,
            |row, col| {
                scale * Spectrum::from_image_rgb(&image.get_pixel(col as u32, row as u32), gamma)
            },
        );

        let log = log.new(o!());

        Self {
            mip_map: MIPMap::new(&log, matrix, true, wrap_mode),
            mapping,
            log,
        }
    }
}

impl ImageTexture<na::Vector3<f32>> {
    pub fn new(
        log: &slog::Logger,
        image: &image::RgbImage,
        scale: na::Vector2<f32>,
        wrap_mode: WrapMode,
        mapping: UVMap,
    ) -> Self {
        let matrix = na::DMatrix::from_fn(
            image.height() as usize,
            image.width() as usize,
            |row, col| {
                let pixel = &image.get_pixel(col as u32, row as u32);
                na::Vector3::new(
                    scale[0] * (pixel[0] as f32 / 127.5 - 1.0),
                    scale[1] * (pixel[1] as f32 / 127.5 - 1.0),
                    pixel[2] as f32 / 127.5 - 1.0,
                )
            },
        );

        let log = log.new(o!());

        Self {
            mip_map: MIPMap::new(&log, matrix, true, wrap_mode),
            mapping,
            log,
        }
    }
}

pub type NormalMap = ImageTexture<na::Vector3<f32>>;

impl<T> Texture<T> for ImageTexture<T>
where
    T: na::Scalar + num::Zero + Copy + AddAssign + Mul<f32, Output = T>,
{
    fn evaluate(&self, it: &SurfaceMediumInteraction) -> T {
        let mut dst_dx = glm::zero();
        let mut dst_dy = glm::zero();
        trace!(self.log, "current mesh uv: {:?}", it.uv);
        let st = self.mapping.map(&it, &mut dst_dx, &mut dst_dy);
        self.mip_map.lookup(&st, &dst_dx, &dst_dy)
    }
}

struct ResampleWeight {
    pub first_texel: usize,
    pub weight: [f32; 4],
}

fn lanczos(mut x: f32, tau: f32) -> f32 {
    x = x.abs();
    if x < 1e-5 {
        return 1.;
    }
    if x > 1. {
        return 0.;
    }
    x *= std::f32::consts::PI;
    let s = (x * tau).sin() / (x * tau);
    let lanczos = x.sin() / x;
    s * lanczos
}

fn resample_weights(old_res: usize, new_res: usize) -> Vec<ResampleWeight> {
    let mut wt = Vec::with_capacity(new_res);
    const FILTER_WIDTH: f32 = 2.;
    for i in 0..new_res {
        let center = (i as f32 + 0.5) * old_res as f32 / new_res as f32;
        let first_texel = ((center - FILTER_WIDTH) + 0.5).floor();
        let mut weight = [0.0; 4];
        for j in 0..4 {
            let pos = first_texel + j as f32 + 0.5;
            weight[j] = lanczos((pos - center) / FILTER_WIDTH, 2.0);
        }

        let inv_sum_wts = 1. / (weight[0] + weight[1] + weight[2] + weight[3]);
        for j in 0..4 {
            weight[j] *= inv_sum_wts;
        }

        wt.push(ResampleWeight {
            first_texel: first_texel as usize,
            weight,
        });
    }
    wt
}

pub struct MIPMap<T: na::Scalar + num::Zero> {
    pyramid: Vec<na::DMatrix<T>>,
    wrap_mode: WrapMode,
    do_trilinear: bool,
    log: slog::Logger,
}

fn texel<T: na::Scalar + num::Zero>(
    level: &na::DMatrix<T>,
    s: i32,
    t: i32,
    wrap_mode: &WrapMode,
) -> T {
    let ret;
    match wrap_mode {
        WrapMode::Repeat => {
            let s = abs_mod(s, level.ncols() as i32);
            let t = abs_mod(t, level.nrows() as i32);
            ret = level[(t as usize, s as usize)].clone();
        }
        WrapMode::Black => {
            if s < 0 || s >= level.ncols() as i32 || t < 0 || t >= level.nrows() as i32 {
                ret = num::zero()
            } else {
                ret = level[(t as usize, s as usize)].clone();
            }
        }
        WrapMode::Clamp => {
            let s = s.clamp(0, level.ncols() as i32 - 1);
            let t = t.clamp(0, level.nrows() as i32 - 1);
            ret = level[(t as usize, s as usize)].clone();
        }
    }

    ret
}

impl<T> MIPMap<T>
where
    T: na::Scalar + num::Zero + Copy + AddAssign + Mul<f32, Output = T>,
{
    pub fn new(
        log: &slog::Logger,
        image: na::DMatrix<T>,
        do_trilinear: bool,
        wrap_mode: WrapMode,
    ) -> Self {
        let log = log.new(o!());
        let mut resolution = na::Point2::new(image.ncols(), image.nrows());
        let resampled_image = if !image.ncols().is_power_of_two()
            || !image.nrows().is_power_of_two()
        {
            debug!(log, "image size not power of two, resampling");
            let res_pow_2 = na::Point2::new(
                round_up_pow_2(image.ncols() as i32) as usize,
                round_up_pow_2(image.nrows() as i32) as usize,
            );

            info!(
                log,
                "Resampling MIPMap from {:?} to {:?}, ratio = {:?}",
                resolution,
                res_pow_2,
                ((res_pow_2.x * res_pow_2.y) / (resolution.x * resolution.y))
            );

            let s_weights = resample_weights(resolution[0], res_pow_2[0]);
            let mut resampled_image = na::DMatrix::<T>::zeros(res_pow_2[1], res_pow_2[0]);

            for t in 0..resolution[1] {
                for s in 0..res_pow_2[0] {
                    for j in 0..4 {
                        let mut orig_s = s_weights[s].first_texel + j;
                        match wrap_mode {
                            WrapMode::Repeat => {
                                orig_s = abs_mod(orig_s, resolution[0]);
                            }
                            WrapMode::Clamp => {
                                orig_s = orig_s.clamp(0, resolution[0] - 1);
                            }
                            _ => {}
                        }
                        if orig_s > 0 && orig_s < resolution[0] {
                            resampled_image[(t, s)] += image[(t, orig_s)] * s_weights[s].weight[j];
                        }
                    }
                }
            }

            let t_weights = resample_weights(resolution[1], res_pow_2[1]);

            for s in 0..res_pow_2[0] {
                let mut work_data = vec![T::zero(); res_pow_2[1]];
                for t in 0..res_pow_2[1] {
                    for j in 0..4 {
                        let mut offset = t_weights[t].first_texel + j;
                        match wrap_mode {
                            WrapMode::Repeat => {
                                offset = abs_mod(offset, resolution[1]);
                            }
                            WrapMode::Clamp => {
                                offset = offset.clamp(0, resolution[1] - 1);
                            }
                            _ => {}
                        }
                        if offset >= 0 && offset < resolution[1] {
                            work_data[t] += resampled_image[(offset, s)] * t_weights[t].weight[j];
                        }
                    }
                }
                for t in 0..res_pow_2[1] {
                    resampled_image[(t, s)] = work_data[t];
                }
            }

            resolution = res_pow_2;

            Some(resampled_image)
        } else {
            None
        };

        let n_levels = 1 + log2_int(resolution[0].max(resolution[1])) as usize;
        let mut pyramid = Vec::with_capacity(n_levels as usize);

        if let Some(resampled_image) = resampled_image {
            pyramid.push(resampled_image);
        } else {
            pyramid.push(image);
        }

        for i in 1..n_levels {
            let s_res = 1usize.max(pyramid[i - 1].ncols() / 2);
            let t_res = 1usize.max(pyramid[i - 1].nrows() / 2);

            pyramid.push(na::DMatrix::from_fn(t_res, s_res, |t, s| {
                (texel(&pyramid[i - 1], (2 * s) as i32, (2 * t) as i32, &wrap_mode)
                    + texel(
                        &pyramid[i - 1],
                        (2 * s + 1) as i32,
                        (2 * t) as i32,
                        &wrap_mode,
                    )
                    + texel(
                        &pyramid[i - 1],
                        (2 * s) as i32,
                        (2 * t + 1) as i32,
                        &wrap_mode,
                    )
                    + texel(
                        &pyramid[i - 1],
                        (2 * s + 1) as i32,
                        (2 * t + 1) as i32,
                        &wrap_mode,
                    ))
                    * 0.25
            }));
        }

        // TODO: EWA filters

        Self {
            pyramid,
            do_trilinear,
            wrap_mode,
            log,
        }
    }

    fn texel(&self, level: usize, s: i32, t: i32) -> T {
        let ret = texel(&self.pyramid[level], s, t, &self.wrap_mode);
        trace!(self.log, "sampled value: {:?}", ret);
        ret
    }

    fn triangle(&self, level: usize, st: &na::Point2<f32>) -> T {
        let level = level.clamp(0, self.pyramid.len() - 1);
        let s = st[0] * self.pyramid[level].ncols() as f32 - 0.5;
        let t = st[1] * self.pyramid[level].nrows() as f32 - 0.5;
        let s0 = s.floor();
        let t0 = t.floor();
        let ds = s - s0;
        let dt = t - t0;
        let s0 = s0 as i32;
        let t0 = t0 as i32;

        self.texel(level, s0, t0) * (1.0 - ds) * (1.0 - dt)
            + self.texel(level, s0, t0 + 1) * (1.0 - ds) * dt
            + self.texel(level, s0 + 1, t0) * ds * (1.0 - dt)
            + self.texel(level, s0 + 1, t0 + 1) * ds * dt
    }

    pub fn lookup(
        &self,
        st: &na::Point2<f32>,
        dst_dx: &na::Vector2<f32>,
        dst_dy: &na::Vector2<f32>,
    ) -> T {
        if self.do_trilinear {
            let width = dst_dx[0]
                .abs()
                .max(dst_dx[1].abs())
                .max(dst_dy[0].abs().max(dst_dy[1].abs()));
            self.lookup_width(&st, width)
        } else {
            panic!("ewa not supported yet!");
        }
    }

    pub fn lookup_width(&self, st: &na::Point2<f32>, width: f32) -> T {
        let level = self.pyramid.len() as f32 - 1.0 + width.max(1e-8).log2();

        if level < 0.0 {
            self.triangle(0, &st)
        } else if level >= (self.pyramid.len() - 1) as f32 {
            self.triangle(self.pyramid.len() - 1, &st)
        } else {
            let i_level = level.floor();
            let delta = level - i_level;
            let i_level = i_level as usize;
            lerp(
                self.triangle(i_level, st),
                self.triangle(i_level + 1, st),
                delta,
            )
        }
    }
}
