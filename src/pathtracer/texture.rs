use super::interaction::SurfaceMediumInteraction;
use crate::common::{spectrum::Spectrum, WrapMode};

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
            mip_map: MIPMap::new(&log, matrix, false, wrap_mode),
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
            mip_map: MIPMap::new(&log, matrix, false, wrap_mode),
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
            mip_map: MIPMap::new(&log, matrix, false, wrap_mode),
            mapping,
            log,
        }
    }
}

pub type NormalMap = ImageTexture<na::Vector3<f32>>;

impl<T: na::Scalar + num::Zero> Texture<T> for ImageTexture<T> {
    fn evaluate(&self, it: &SurfaceMediumInteraction) -> T {
        let mut dst_dx = glm::zero();
        let mut dst_dy = glm::zero();
        trace!(self.log, "current mesh uv: {:?}", it.uv);
        let st = self.mapping.map(&it, &mut dst_dx, &mut dst_dy);
        self.mip_map.lookup(&st, &dst_dx, &dst_dy)
    }
}

pub struct MIPMap<T: na::Scalar + num::Zero> {
    pyramid: Vec<na::DMatrix<T>>,
    wrap_mode: WrapMode,
    do_trilinear: bool,
    resolution: na::Point2<i32>,
    log: slog::Logger,
}

impl<T: na::Scalar + num::Zero> MIPMap<T> {
    pub fn new(
        log: &slog::Logger,
        image: na::DMatrix<T>,
        do_trilinear: bool,
        wrap_mode: WrapMode,
    ) -> Self {
        let log = log.new(o!());
        Self {
            resolution: na::Point2::new(image.ncols() as i32, image.nrows() as i32),
            pyramid: vec![image],
            do_trilinear,
            wrap_mode,
            log,
        }
    }

    fn texel(&self, level: usize, s: i32, t: i32) -> T {
        let ret;
        let level = &self.pyramid[level];
        match self.wrap_mode {
            WrapMode::Repeat => {
                let mut s = s % level.ncols() as i32;
                s = if s < 0 { s + level.ncols() as i32 } else { s };
                let mut t = t % level.nrows() as i32;
                t = if t < 0 { t + level.nrows() as i32 } else { t };
                trace!(self.log, "[Repeat] processed: {:?}, {:?}", s, t);
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
        trace!(self.log, "sampled value: {:?}", ret);

        ret
    }

    fn triangle(&self, level: usize, st: &na::Point2<f32>) -> T {
        let level = level.clamp(0, self.pyramid.len() - 1);
        let s = st[0] * self.pyramid[level].ncols() as f32 - 0.5;
        let t = st[1] * self.pyramid[level].nrows() as f32 - 0.5;
        let s0 = s.floor() as i32;
        let t0 = t.floor() as i32;

        self.texel(level, s0, t0)
    }

    pub fn lookup(
        &self,
        st: &na::Point2<f32>,
        dst_dx: &na::Vector2<f32>,
        dst_dy: &na::Vector2<f32>,
    ) -> T {
        // TODO: do more sophisticated texture handling
        if self.do_trilinear {
            self.triangle(0, &st)
        } else {
            self.triangle(0, &st)
        }
    }

    pub fn lookup_width(&self, st: &na::Point2<f32>, width: f32) -> T {
        self.triangle(0, &st)
    }
}
