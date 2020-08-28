use std::cell::RefCell;

#[derive(Clone, Debug)]
pub struct Ray {
    pub o: na::Point3<f32>,
    pub d: na::Vector3<f32>,
    pub t_max: RefCell<f32>,
}

impl Ray {
    pub fn point_at(&self, t: f32) -> na::Point3<f32> {
        self.o + self.d * t
    }
}
