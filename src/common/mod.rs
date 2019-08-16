use nalgebra::{Perspective3, Projective3, Transform3, Vector3};

pub struct Camera {
    pub view: Projective3<f32>,
    pub projection: Perspective3<f32>,
}

pub struct Mesh {
    pub vertex_indices: Vec<u32>,
    pub pos: Vec<Vector3<f32>>,
    pub normal: Vec<Vector3<f32>>,
}

pub struct World {
    pub camera: Camera,
    pub object_transforms: Vec<Transform3<f32>>,
}
