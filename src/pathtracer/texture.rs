use super::interaction::SurfaceInteraction;

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
