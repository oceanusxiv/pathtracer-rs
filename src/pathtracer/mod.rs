mod material;
mod primitive;
mod sampling;
mod shape;

use crate::common::{Camera, World};
use image::RgbImage;
use indicatif::ProgressBar;
use material::Material;
use std::cell::RefCell;

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

pub struct TBounds3<T: glm::Scalar> {
    pub p_min: glm::TVec3<T>,
    pub p_max: glm::TVec3<T>,
}

pub type Bounds3 = TBounds3<f32>;

pub struct SurfaceInteraction {
    pub p: glm::Vec3,
    pub p_error: glm::Vec3,
    pub wo: glm::Vec3,
    pub n: glm::Vec3,
}

pub trait Sampler {}

impl Camera {
    pub fn generate_ray(&self, film_point: glm::Vec2) -> Ray {
        let mut cam_dir = self.cam_to_screen.unproject_point(&(self.raster_to_screen * na::Point3::new(film_point.x, film_point.y, 0.0)));

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
        let mut primitives: Vec<Box<dyn primitive::Primitive>> = Vec::new();

        for obj in &world.objects {
            for shape in obj.mesh.to_shapes(&obj) {
                primitives.push(Box::new(primitive::GeometricPrimitive { shape }))
            }
        }

        RenderScene {
            scene: Box::new(primitive::Aggregate::new(primitives)),
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
        println!("{:?}", &camera.cam_to_world.translation);
        let mut intersections = 0;
        let bar = ProgressBar::new((film.image.width() * film.image.height()) as u64);
        for (x, y, pixel) in film.image.enumerate_pixels_mut() {
            let ray = camera.generate_ray(glm::vec2(x as f32, y as f32) + glm::vec2(0.5, 0.5));
            if let Some(isect) = scene.scene.intersect(&ray) {
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
