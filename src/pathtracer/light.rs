use crate::common::{ray::Ray, spectrum::Spectrum};

pub trait Light {
    fn le(&self, r: &Ray) -> Spectrum {
        Spectrum::new(0.0)
    }
}

pub trait SyncLight: Light + Send + Sync {}
impl<T> SyncLight for T where T: Light + Send + Sync {}

pub struct InfiniteAreaLight {
    L: Spectrum,
}
