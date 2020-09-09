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
    resolution: na::Point2<f32>,
    log: slog::Logger,
}

impl<T: na::Scalar + num::Zero> MIPMap<T> {
    fn new(
        log: &slog::Logger,
        image: na::DMatrix<T>,
        do_trilinear: bool,
        wrap_mode: WrapMode,
    ) -> Self {
        let log = log.new(o!());
        Self {
            resolution: na::Point2::new(image.ncols() as f32, image.nrows() as f32),
            pyramid: vec![image],
            wrap_mode,
            log,
        }
    }

    fn lookup(
        &self,
        st: &na::Point2<f32>,
        dst_dx: &na::Vector2<f32>,
        dst_dy: &na::Vector2<f32>,
    ) -> T {
        let ret;
        match self.wrap_mode {
            WrapMode::Repeat => {
                let mut s = st[0] % self.resolution[0];
                s = if s < 0.0 { s + self.resolution[0] } else { s };
                let mut t = st[1] % self.resolution[1];
                t = if t < 0.0 { t + self.resolution[1] } else { t };
                trace!(
                    self.log,
                    "[Repeat] original: {:?}, {:?}, processed: {:?}, {:?}",
                    st[0],
                    st[1],
                    s,
                    t
                );
                ret = self.pyramid[0][(
                    t.floor().clamp(0.0, self.resolution.y - 1.0) as usize,
                    s.floor().clamp(0.0, self.resolution.x - 1.0) as usize,
                )]
                    .clone();
            }
            WrapMode::Black => ret = num::zero(),
            WrapMode::Clamp => {
                let s = st[0].clamp(0.0, self.resolution[0]);
                let t = st[1].clamp(0.0, self.resolution[1]);
                ret = self.pyramid[0][(t.floor() as usize, s.floor() as usize)].clone();
            }
        }
        trace!(self.log, "sampled value: {:?}", ret);

        ret
    }
}
