use super::interaction::SurfaceInteraction;
use crate::common::{spectrum::Spectrum, WrapMode};

pub trait Texture<T> {
    fn evaluate(&self, it: &SurfaceInteraction) -> T;
}

pub trait SyncTexture<T>: Texture<T> + Send + Sync {}
impl<T1, T2> SyncTexture<T1> for T2 where T2: Texture<T1> + Send + Sync {}

pub struct ConstantTexture<T: Copy> {
    value: T,
}

impl<T: Copy> Texture<T> for ConstantTexture<T> {
    fn evaluate(&self, it: &SurfaceInteraction) -> T {
        self.value
    }
}

fn uv_map(
    it: &SurfaceInteraction,
    dst_dx: &mut na::Vector2<f32>,
    dst_dy: &mut na::Vector2<f32>,
) -> na::Point2<f32> {
    *dst_dx = na::Vector2::new(it.dudx, it.dvdx);
    *dst_dy = na::Vector2::new(it.dudy, it.dvdy);

    it.uv
}

pub struct ImageTexture<T: na::Scalar> {
    mip_map: MIPMap<T>,
}

impl ImageTexture<f32> {
    pub fn new(image: image::GrayImage, scale: f32) -> Self {
        let matrix = na::DMatrix::<f32>::from_fn(
            image.height() as usize,
            image.width() as usize,
            |row, col| scale * (image.get_pixel(col as u32, row as u32)[0] as f32 / 255.0),
        );

        Self {
            mip_map: MIPMap::new(matrix, false, WrapMode::Clamp),
        }
    }
}

impl ImageTexture<Spectrum> {
    pub fn new(image: image::RgbImage, scale: Spectrum) -> Self {
        let matrix = na::DMatrix::<Spectrum>::from_fn(
            image.height() as usize,
            image.width() as usize,
            |row, col| Spectrum::from_image_rgb(&image.get_pixel(col as u32, row as u32)),
        );

        Self {
            mip_map: MIPMap::new(matrix, false, WrapMode::Clamp),
        }
    }
}

impl<T: na::Scalar> Texture<T> for ImageTexture<T> {
    fn evaluate(&self, it: &SurfaceInteraction) -> T {
        let mut dst_dx = glm::zero();
        let mut dst_dy = glm::zero();
        let st = uv_map(&it, &mut dst_dx, &mut dst_dy);
        self.mip_map.lookup(&st, &dst_dx, &dst_dy)
    }
}

struct MIPMap<T: na::Scalar> {
    pyramid: Vec<na::DMatrix<T>>,
}

impl<T: na::Scalar> MIPMap<T> {
    fn new(image: na::DMatrix<T>, do_trilinear: bool, wrap_mode: WrapMode) -> Self {
        Self {
            pyramid: vec![image],
        }
    }

    fn lookup(
        &self,
        st: &na::Point2<f32>,
        dst_dx: &na::Vector2<f32>,
        dst_dy: &na::Vector2<f32>,
    ) -> T {
        self.pyramid[0][(st.x.floor() as usize, st.y.floor() as usize)].clone()
    }
}
