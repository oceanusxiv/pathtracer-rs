pub trait Sampler {
    fn get_1d(&self) -> f32;
    fn get_2d(&self) -> na::Point2<i32>;
}

fn uniform_sample_hemisphere(u: &glm::Vec2) -> glm::Vec3 {
    glm::vec3(0.0, 0.0, 0.0)
}

pub fn concentric_sample_disk(u: &na::Point2<f32>) -> na::Point2<f32> {
    let u_offset = 2.0 * u - na::Vector2::new(1.0, 1.0);

    if u_offset.x == 0.0 && u_offset.y == 0.0 {
        na::Point2::new(0.0, 0.0)
    } else {
        let mut theta = 0.0f32;
        let mut r = 0.0f32;

        if u_offset.x.abs() > u_offset.y.abs() {
            r = u_offset.x;
            theta = std::f32::consts::FRAC_PI_4 * (u_offset.y / u_offset.x);
        } else {
            r = u_offset.y;
            theta = std::f32::consts::FRAC_PI_2
                - std::f32::consts::FRAC_PI_4 * (u_offset.x / u_offset.y);
        }

        r * na::Point2::new(theta.cos(), theta.sin())
    }
}

pub fn cosine_sample_hemisphere(u: &na::Point2<f32>) -> na::Vector3<f32> {
    let d = concentric_sample_disk(&u);
    let z = 0.0f32.max(1.0 - d.x * d.x - d.y * d.y).sqrt();
    na::Vector3::new(d.x, d.y, z)
}

pub fn cosine_hemisphere_pdf(cos_theta: f32) -> f32 {
    cos_theta * std::f32::consts::FRAC_1_PI
}
