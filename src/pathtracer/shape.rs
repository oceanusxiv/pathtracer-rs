use std::rc::Rc;

pub trait Shape {
    fn intersect() {}
    fn area() {}
}

use crate::common::Mesh;
pub struct Triangle {
    mesh: Rc<Mesh>,
    indices: [u32; 3],
}

impl Triangle {}

impl Shape for Triangle {}
