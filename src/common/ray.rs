use std::cell::RefCell;

#[derive(Clone, Debug)]
pub struct Ray {
    pub o: na::Point3<f32>,
    pub d: na::Vector3<f32>,
    pub t_max: RefCell<f32>,
}

#[derive(Clone, Debug)]
pub struct RayDifferential {
    pub ray: Ray,
    pub has_differentials: bool,
    pub rx_origin: na::Point3<f32>,
    pub ry_origin: na::Point3<f32>,
    pub rx_direction: na::Vector3<f32>,
    pub ry_direction: na::Vector3<f32>,
}

impl RayDifferential {
    pub fn new(ray: Ray) -> Self {
        Self {
            ray,
            has_differentials: false,
            rx_origin: na::Point3::origin(),
            ry_origin: na::Point3::origin(),
            rx_direction: glm::zero(),
            ry_direction: glm::zero(),
        }
    }

    pub fn scale_differentials(&mut self, s: f32) {
        self.rx_origin = self.ray.o + (self.rx_origin - self.ray.o) * s;
        self.ry_origin = self.ray.o + (self.ry_origin - self.ray.o) * s;
        self.rx_direction = self.ray.d + (self.rx_direction - self.ray.d) * s;
        self.ry_direction = self.ray.d + (self.ry_direction - self.ray.d) * s;
    }
}
