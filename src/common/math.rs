const MACHINE_EPSILON: f32 = f32::EPSILON * 0.5;
pub const INV_2_PI: f32 = std::f32::consts::FRAC_1_PI * 0.5;

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

pub fn face_forward<T: na::RealField + num::FromPrimitive>(
    n: &na::Vector3<T>,
    v: &na::Vector3<T>,
) -> na::Vector3<T> {
    if n.dot(v) < T::from_f32(0.0).unwrap() {
        -n
    } else {
        *n
    }
}

pub fn coordinate_system<T: na::RealField + num::FromPrimitive>(
    v1: &na::Vector3<T>,
    v2: &mut na::Vector3<T>,
    v3: &mut na::Vector3<T>,
) {
    if v1.x.abs() > v1.y.abs() {
        *v2 = na::Vector3::new(-v1.z, T::from_f32(0.0).unwrap(), v1.x)
            / (v1.x * v1.x + v1.z * v1.z).sqrt();
    } else {
        *v2 = na::Vector3::new(T::from_f32(0.0).unwrap(), v1.z, -v1.y)
            / (v1.y * v1.y + v1.z * v1.z).sqrt();
    }
    *v3 = v1.cross(v2);
}

pub fn float_to_bits(f: f32) -> u32 {
    unsafe { std::mem::transmute::<f32, u32>(f) }
}

pub fn bits_to_float(f: u32) -> f32 {
    unsafe { std::mem::transmute::<u32, f32>(f) }
}

pub fn next_float_up(mut v: f32) -> f32 {
    // Handle infinity and negative zero for _NextFloatUp()_
    if v.is_infinite() && v > 0. {
        return v;
    }
    if v == -0.0 {
        v = 0.0;
    }

    // Advance _v_ to next higher float
    let mut ui = float_to_bits(v);
    if v >= 0.0 {
        ui += 1;
    } else {
        ui -= 1;
    }
    return bits_to_float(ui);
}

pub fn next_float_down(mut v: f32) -> f32 {
    // Handle infinity and positive zero for _NextFloatDown()_
    if v.is_infinite() && v < 0.0 {
        return v;
    }
    if v == 0.0 {
        v = -0.0;
    }
    let mut ui = float_to_bits(v);
    if v > 0.0 {
        ui += 1;
    } else {
        ui -= 1;
    }
    return bits_to_float(ui);
}

pub fn offset_ray_origin(
    p: &na::Point3<f32>,
    p_error: &na::Vector3<f32>,
    n: &na::Vector3<f32>,
    w: &na::Vector3<f32>,
) -> na::Point3<f32> {
    let d = n.abs().dot(p_error);
    let mut offset = d * n;

    if w.dot(n) < 0.0 {
        offset = -offset;
    }

    let mut po = p + offset;

    for i in 0..3 {
        if offset[i] > 0.0 {
            po[i] = next_float_up(po[i]);
        } else if offset[i] < 0.0 {
            po[i] = next_float_down(po[i]);
        }
    }

    return po;
}

pub fn gamma_correct(value: f32) -> f32 {
    if value <= 0.0031308f32 {
        return 12.92 * value;
    }

    1.055 * value.powf(1.0 / 2.4) - 0.055
}

pub fn inverse_gamma_correct(value: f32) -> f32 {
    if value <= 0.04045 {
        return value * 1.0 / 12.92;
    }

    ((value + 0.055) * 1.0 / 1.055).powf(2.4)
}

pub fn solve_linear_system_2x2(
    a: &na::Matrix2<f32>,
    b: &na::Vector2<f32>,
) -> Option<na::Vector2<f32>> {
    let det = a.determinant();
    if det.abs() < 1e-10f32 {
        return None;
    }
    let x0 = (a[(1, 1)] * b[0] - a[(0, 1)] * b[1]) / det;
    let x1 = (a[(0, 0)] * b[1] - a[(1, 0)] * b[0]) / det;

    if x0.is_nan() || x1.is_nan() {
        return None;
    }

    return Some(na::Vector2::new(x0, x1));
}

pub fn power_heuristic(nf: i32, f_pdf: f32, ng: i32, g_pdf: f32) -> f32 {
    let f = nf as f32 * f_pdf;
    let g = ng as f32 * g_pdf;
    (f * f) / (f * f + g * g)
}

pub fn spherical_theta(v: &na::Vector3<f32>) -> f32 {
    v.z.clamp(-1.0, 1.0).acos()
}

pub fn spherical_phi(v: &na::Vector3<f32>) -> f32 {
    let p = v.y.atan2(v.x);
    if p < 0.0 {
        p + 2.0 * std::f32::consts::PI
    } else {
        p
    }
}

pub fn find_interval<T: Fn(usize) -> bool>(size: usize, pred: T) -> usize {
    let mut first = 0;
    let mut len = size;

    while len > 0 {
        let half = len >> 1;
        let middle = first + half;
        if pred(middle) {
            first = middle + 1;
            len -= half + 1;
        } else {
            len = half;
        }
    }

    (first - 1).clamp(0, size - 2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solve_linear_system_2x2() {
        let a = na::Matrix2::new(0.0, 1.0, 1.0, 0.0);
        let b = na::Vector2::new(2.0, 4.0);

        let x = solve_linear_system_2x2(&a, &b);
        assert!(x.is_some());
        let x = x.unwrap();
        assert_eq!(x, glm::vec2(4.0, 2.0));

        let a = na::Matrix2::new(0.0, 0.0, 0.0, 0.0);
        let b = na::Vector2::new(2.0, 4.0);

        assert!(solve_linear_system_2x2(&a, &b).is_none());

        let a = na::Matrix2::new(1.0, 1.0, -1.0, 1.0);
        let b = na::Vector2::new(2.0, 2.0);

        let x = solve_linear_system_2x2(&a, &b);
        assert!(x.is_some());
        let x = x.unwrap();
        assert_eq!(x, glm::vec2(0.0, 2.0));
    }
}
