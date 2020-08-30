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