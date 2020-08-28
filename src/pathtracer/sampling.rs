pub trait Sampler {
    fn get_1d(&self) -> f32;
    fn get_2d(&self) -> na::Point2<i32>;
}

fn uniform_sample_hemisphere(u: &glm::Vec2) -> glm::Vec3 {
    glm::vec3(0.0, 0.0, 0.0)
}
