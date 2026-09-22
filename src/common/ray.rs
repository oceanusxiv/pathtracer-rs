#[derive(Clone, Debug)]
pub struct TRay<T: simba::simd::SimdRealField> {
    pub o: na::Point3<T>,
    pub d: na::Vector3<T>,
    pub t_max: T,
}

pub type Ray = TRay<f32>;

#[derive(Clone, Debug)]
pub struct TRayDifferential<TF: simba::simd::SimdRealField> {
    pub ray: TRay<TF>,
    pub has_differentials: bool,
    pub rx_origin: na::Point3<TF>,
    pub ry_origin: na::Point3<TF>,
    pub rx_direction: na::Vector3<TF>,
    pub ry_direction: na::Vector3<TF>,
}

pub type RayDifferential = TRayDifferential<f32>;

impl<TF: simba::simd::SimdRealField + Copy> TRayDifferential<TF> {
    pub fn new(ray: TRay<TF>) -> Self {
        Self {
            ray,
            has_differentials: false,
            rx_origin: na::Point3::origin(),
            ry_origin: na::Point3::origin(),
            rx_direction: glm::zero(),
            ry_direction: glm::zero(),
        }
    }

    pub fn scale_differentials(&mut self, s: TF) {
        self.rx_origin = self.ray.o + (self.rx_origin - self.ray.o) * s;
        self.ry_origin = self.ray.o + (self.ry_origin - self.ray.o) * s;
        self.rx_direction = self.ray.d + (self.rx_direction - self.ray.d) * s;
        self.ry_direction = self.ray.d + (self.ry_direction - self.ray.d) * s;
    }
}
