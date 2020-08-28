use super::math::*;
use super::ray::Ray;

#[derive(Debug, Clone, Copy)]
pub struct TBounds2<T: na::Scalar> {
    pub p_min: na::Point2<T>,
    pub p_max: na::Point2<T>,
}

pub type Bounds2i = TBounds2<i32>;

impl<T: na::Scalar + na::ClosedSub + na::ClosedMul + Copy> TBounds2<T> {
    pub fn diagonal(&self) -> na::Vector2<T> {
        self.p_max.coords - self.p_min.coords
    }

    pub fn area(&self) -> T {
        let d = self.p_max.coords - self.p_min.coords;
        d.x * d.y
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TBounds3<T: na::RealField> {
    pub p_min: na::Point3<T>,
    pub p_max: na::Point3<T>,
}

pub fn min_p<T: na::RealField>(p1: &na::Point3<T>, p2: &na::Point3<T>) -> na::Point3<T> {
    na::Point3::new(
        na::RealField::min(p1.x, p2.x),
        na::RealField::min(p1.y, p2.y),
        na::RealField::min(p1.z, p2.z),
    )
}

pub fn max_p<T: na::RealField>(p1: &na::Point3<T>, p2: &na::Point3<T>) -> na::Point3<T> {
    na::Point3::new(
        na::RealField::max(p1.x, p2.x),
        na::RealField::max(p1.y, p2.y),
        na::RealField::max(p1.z, p2.z),
    )
}

impl<T: na::RealField> TBounds3<T> {
    pub fn new(p1: na::Point3<T>, p2: na::Point3<T>) -> Self {
        TBounds3 {
            p_min: min_p(&p1, &p2),
            p_max: max_p(&p1, &p2),
        }
    }
}

pub type Bounds3 = TBounds3<f32>;

impl<T: na::RealField + na::ClosedSub> TBounds3<T> {
    pub fn empty() -> Self {
        let min_num = T::min_value();
        let max_num = T::max_value();

        TBounds3 {
            p_min: na::Point3::new(max_num, max_num, max_num),
            p_max: na::Point3::new(min_num, min_num, min_num),
        }
    }

    pub fn diagonal(&self) -> na::Vector3<T> {
        self.p_max.coords - self.p_min.coords
    }

    pub fn maximum_extent(&self) -> usize {
        self.diagonal().imax()
    }

    pub fn offset(&self, p: &na::Point3<T>) -> na::Vector3<T> {
        let mut o = p - self.p_min;
        if self.p_max.x > self.p_min.x {
            o.x /= self.p_max.x - self.p_min.x;
        }
        if self.p_max.y > self.p_min.y {
            o.y /= self.p_max.y - self.p_min.y;
        }
        if self.p_max.z > self.p_min.z {
            o.z /= self.p_max.z - self.p_min.z;
        }

        o
    }

    pub fn surface_area(&self) -> T {
        let d = self.diagonal();
        T::from_f64(2.0).unwrap() * (d.x * d.y + d.x * d.z + d.y * d.z)
    }
}

impl<T: na::RealField> std::ops::Index<usize> for TBounds3<T> {
    type Output = na::Point3<T>;

    fn index(&self, i: usize) -> &Self::Output {
        if i == 0 {
            &self.p_min
        } else {
            &self.p_max
        }
    }
}

impl<T: na::RealField> TBounds3<T> {
    pub fn union(b1: &TBounds3<T>, b2: &TBounds3<T>) -> TBounds3<T> {
        TBounds3 {
            p_min: min_p(&b1.p_min, &b2.p_min),
            p_max: max_p(&b1.p_max, &b2.p_max),
        }
    }

    pub fn union_p(b: &TBounds3<T>, p: &na::Point3<T>) -> TBounds3<T> {
        TBounds3 {
            p_min: min_p(&b.p_min, &p),
            p_max: max_p(&b.p_max, &p),
        }
    }
}

impl Bounds3 {
    pub fn intersect_p(&self, r: &Ray) -> Option<(f32, f32)> {
        let mut t0 = 0.0;
        let mut t1 = *r.t_max.borrow();

        for i in 0..3usize {
            let inv_ray_dir: f32 = 1.0 / r.d[i];
            let mut t_near: f32 = (self.p_min[i] - r.o[i]) * inv_ray_dir;
            let mut t_far: f32 = (self.p_max[i] - r.o[i]) * inv_ray_dir;

            if t_near > t_far {
                std::mem::swap(&mut t_near, &mut t_far);
            }

            t_far *= 1.0 + 2.0 * gamma(3);
            t0 = if t_near > t0 { t_near } else { t0 };
            t1 = if t_far < t1 { t_far } else { t1 };
            if t0 > t1 {
                return None;
            }
        }

        Some((t0, t1))
    }

    pub fn intersect_p_precomp(
        &self,
        r: &Ray,
        inv_dir: &na::Vector3<f32>,
        dir_is_neg: &[bool; 3],
    ) -> bool {
        // Check for ray intersection against $x$ and $y$ slabs
        let mut t_min = (self[dir_is_neg[0] as usize].x - r.o.x) * inv_dir.x;
        let mut t_max = (self[1 - dir_is_neg[0] as usize].x - r.o.x) * inv_dir.x;
        let ty_min = (self[dir_is_neg[1] as usize].y - r.o.y) * inv_dir.y;
        let mut ty_max = (self[1 - dir_is_neg[1] as usize].y - r.o.y) * inv_dir.y;

        // Update _tMax_ and _tyMax_ to ensure robust bounds intersection
        t_max *= 1.0 + 2.0 * gamma(3);
        ty_max *= 1.0 + 2.0 * gamma(3);
        if t_min > ty_max || ty_min > t_max {
            return false;
        };
        if ty_min > t_min {
            t_min = ty_min
        };
        if ty_max < t_max {
            t_max = ty_max
        };

        // Check for ray intersection against $z$ slab
        let tz_min = (self[dir_is_neg[2] as usize].z - r.o.z) * inv_dir.z;
        let mut tz_max = (self[1 - dir_is_neg[2] as usize].z - r.o.z) * inv_dir.z;

        // Update _tzMax_ to ensure robust bounds intersection
        tz_max *= 1.0 + 2.0 * gamma(3);
        if t_min > tz_max || tz_min > t_max {
            return false;
        };
        if tz_min > t_min {
            t_min = tz_min
        };
        if tz_max < t_max {
            t_max = tz_max
        };

        (t_min < *r.t_max.borrow()) && (t_max > 0.0)
    }
}
