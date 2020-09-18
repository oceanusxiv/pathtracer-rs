use std::sync::Arc;

use super::texture::SyncTexture;

pub struct TriangleMesh {
    pub indices: Vec<u32>,
    pub pos: Vec<na::Point3<f32>>,
    pub normal: Vec<na::Vector3<f32>>,
    pub s: Vec<na::Vector3<f32>>,
    pub uv: Vec<na::Point2<f32>>,
    pub colors: Vec<na::Vector3<f32>>,
    pub alpha_mask: Option<Arc<dyn SyncTexture<f32>>>,
}
