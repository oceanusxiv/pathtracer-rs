const MACHINE_EPSILON: f32 = std::f32::EPSILON * 0.5;

pub fn gamma(n: u32) -> f32 {
    (n as f32 * MACHINE_EPSILON) / (1.0 - n as f32 * MACHINE_EPSILON)
}

pub fn max_dimension<N: glm::Scalar + std::cmp::PartialOrd>(v: &glm::TVec3<N>) -> usize {
    if v.x > v.y {
        if v.x > v.z {
            0
        } else {
            2
        }
    } else {
        if v.y > v.z {
            1
        } else {
            2
        }
    }
}

pub fn permute<N: glm::Scalar + std::marker::Copy>(
    p: &glm::TVec3<N>,
    x: usize,
    y: usize,
    z: usize,
) -> glm::TVec3<N> {
    glm::vec3(p[x], p[y], p[z])
}
