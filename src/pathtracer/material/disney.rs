use crate::{common::spectrum::Spectrum, pathtracer::texture::SyncTexture};

pub struct DisneyMaterial {
    color: Box<dyn SyncTexture<Spectrum>>,
    metallic: Box<dyn SyncTexture<f32>>,
    eta: Box<dyn SyncTexture<f32>>,
    roughness: Box<dyn SyncTexture<f32>>,
    normal_map: Option<Box<dyn SyncTexture<na::Vector3<f32>>>>,
}

impl DisneyMaterial {
    // pub fn new() -> Self {

    // }
}
