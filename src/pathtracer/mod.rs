mod accelerator;
mod material;
mod primitive;
mod sampling;
mod shape;

use crate::common::{max_p, min_p, Bounds3, Camera, TBounds3, World};
use image::RgbImage;
use indicatif::ProgressBar;
use material::Material;
use std::cell::RefCell;
use std::rc::Rc;

const MACHINE_EPSILON: f32 = std::f32::EPSILON * 0.5;

fn gamma(n: u32) -> f32 {
    (n as f32 * MACHINE_EPSILON) / (1.0 - n as f32 * MACHINE_EPSILON)
}

fn max_dimension<N: glm::Scalar + std::cmp::PartialOrd>(v: &glm::TVec3<N>) -> usize {
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

fn permute<N: glm::Scalar + std::marker::Copy>(
    p: &glm::TVec3<N>,
    x: usize,
    y: usize,
    z: usize,
) -> glm::TVec3<N> {
    glm::vec3(p[x], p[y], p[z])
}

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

impl<T: na::RealField + na::ClosedSub + num::FromPrimitive> TBounds3<T> {
    fn empty() -> Self {
        let min_num = T::min_value();
        let max_num = T::max_value();

        TBounds3 {
            p_min: na::Point3::new(max_num, max_num, max_num),
            p_max: na::Point3::new(min_num, min_num, min_num),
        }
    }

    fn diagonal(&self) -> na::Vector3<T> {
        self.p_max.coords - self.p_min.coords
    }

    fn maximum_extent(&self) -> usize {
        self.diagonal().imax()
    }

    fn offset(&self, p: &na::Point3<T>) -> na::Vector3<T> {
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

    fn surface_area(&self) -> T {
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

pub struct SurfaceInteraction {
    pub p: glm::Vec3,
    pub p_error: glm::Vec3,
    pub wo: glm::Vec3,
    pub n: glm::Vec3,
}

impl SurfaceInteraction {
    pub fn new() -> Self {
        SurfaceInteraction {
            p: glm::zero(),
            p_error: glm::zero(),
            wo: glm::zero(),
            n: glm::zero(),
        }
    }
}

pub trait Sampler {}

impl Camera {
    pub fn generate_ray(&self, film_point: glm::Vec2) -> Ray {
        let mut cam_dir = self.cam_to_screen.unproject_point(
            &(self.raster_to_screen * na::Point3::new(film_point.x, film_point.y, 0.0)),
        );

        let cam_orig = na::Point3::<f32>::new(0.0, 0.0, 0.0);
        let world_orig = self.cam_to_world * cam_orig;
        let world_dir = self.cam_to_world * cam_dir.coords;
        Ray {
            o: world_orig,
            d: world_dir.normalize(),
            t_max: RefCell::new(std::f32::INFINITY),
        }
    }
}

pub struct Film {
    pub image: Box<RgbImage>,
}

impl Film {
    pub fn new(resolution: &glm::UVec2) -> Self {
        Film {
            image: Box::new(RgbImage::new(resolution.x, resolution.y)),
        }
    }

    pub fn save(&self, file_path: &str) {
        self.image.save(file_path).unwrap()
    }
}

pub struct RenderScene {
    scene: Box<dyn primitive::Primitive>,
}

impl RenderScene {
    pub fn from_world(world: &World) -> Self {
        let mut primitives: Vec<Rc<dyn primitive::Primitive>> = Vec::new();

        for obj in &world.objects {
            for shape in obj.mesh.to_shapes(&obj) {
                primitives.push(Rc::new(primitive::GeometricPrimitive { shape }))
            }
        }

        RenderScene {
            scene: Box::new(accelerator::BVH::new(primitives, &4)),
        }
    }
}

pub trait Integrator {}

pub struct DirectLightingIntegrator {}

impl DirectLightingIntegrator {
    pub fn new() -> Self {
        DirectLightingIntegrator {}
    }

    pub fn render(&self, camera: &Camera, scene: &RenderScene, out_path: &str) {
        let mut film = Film::new(&glm::vec2(
            super::common::DEFAULT_RESOLUTION.x as u32,
            super::common::DEFAULT_RESOLUTION.y as u32,
        ));

        println!(
            "start rendering image of size: {:?} x {:?}",
            film.image.width(),
            film.image.height()
        );
        let mut intersections = 0;
        let bar = ProgressBar::new((film.image.width() * film.image.height()) as u64);
        for (x, y, pixel) in film.image.enumerate_pixels_mut() {
            let ray = camera.generate_ray(glm::vec2(x as f32, y as f32) + glm::vec2(0.5, 0.5));
            let mut isect = SurfaceInteraction::new();
            if scene.scene.intersect(&ray, &mut isect) {
                *pixel = image::Rgb([255u8, 255u8, 255u8]);
                intersections += 1;
            }
            bar.inc(1);
        }
        bar.finish();
        println!("{:?} collisions", intersections);
        println!("saving image to {:?}", out_path);
        film.save(out_path);
    }
}

impl Integrator for DirectLightingIntegrator {}
