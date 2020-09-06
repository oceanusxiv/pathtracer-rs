mod accelerator;
mod bsdf;
mod bxdf;
pub mod integrator;
mod interaction;
pub mod light;
mod material;
mod primitive;
pub mod sampling;
mod shape;
mod texture;

use crate::common::bounds::Bounds3;
use crate::common::ray::{Ray, RayDifferential};
use crate::common::spectrum::Spectrum;
use crate::common::{Camera, World};
use interaction::SurfaceInteraction;
use light::{DiffuseAreaLight, DirectionalLight, Light, PointLight, SyncLight};
use material::{Material, MaterialInterface};
use primitive::SyncPrimitive;
use shape::shape_from_mesh;
use std::cell::RefCell;
use std::sync::Arc;

pub enum TransportMode {
    Radiance,
    Importance,
}

#[derive(Debug)]
pub struct CameraSample {
    p_film: na::Point2<f32>,
}

impl Camera {
    pub fn generate_ray(&self, sample: &CameraSample) -> Ray {
        let p_camera = self.cam_to_screen.unproject_point(
            &(self.raster_to_screen * na::Point3::new(sample.p_film.x, sample.p_film.y, 0.0)),
        );

        let cam_orig = na::Point3::<f32>::new(0.0, 0.0, 0.0);
        let world_orig = self.cam_to_world * cam_orig;
        let world_dir = self.cam_to_world * p_camera.coords;
        Ray {
            o: world_orig,
            d: world_dir.normalize(),
            t_max: RefCell::new(f32::INFINITY),
        }
    }

    pub fn generate_ray_differential(&self, sample: &CameraSample) -> RayDifferential {
        let p_camera = self.cam_to_screen.unproject_point(
            &(self.raster_to_screen * na::Point3::new(sample.p_film.x, sample.p_film.y, 0.0)),
        );

        let cam_orig = na::Point3::<f32>::new(0.0, 0.0, 0.0);
        let world_orig = self.cam_to_world * cam_orig;
        let world_dir = self.cam_to_world * p_camera.coords;
        let rx_world_dir = self.cam_to_world * (p_camera.coords + self.dx_camera);
        let ry_world_dir = self.cam_to_world * (p_camera.coords + self.dy_camera);
        RayDifferential {
            ray: Ray {
                o: world_orig,
                d: world_dir.normalize(),
                t_max: RefCell::new(f32::INFINITY),
            },
            has_differentials: true,
            rx_origin: world_orig,
            ry_origin: world_orig,
            rx_direction: rx_world_dir.normalize(),
            ry_direction: ry_world_dir.normalize(),
        }
    }
}

pub struct RenderScene {
    scene: Box<dyn SyncPrimitive>,
    pub lights: Vec<Arc<dyn SyncLight>>,
    materials: Vec<Arc<Material>>,
    world_bound: Bounds3,
}

impl RenderScene {
    pub fn from_world(log: &slog::Logger, world: &World) -> Self {
        let mut primitives: Vec<Arc<dyn SyncPrimitive>> = Vec::new();
        let mut materials = Vec::new();
        let mut lights: Vec<Arc<dyn SyncLight>> = Vec::new();

        for mat in &world.materials {
            materials.push(Arc::new(Material::from_gltf(log, &**mat)));
        }

        for obj in &world.objects {
            for shape in shape_from_mesh(
                log,
                &obj.mesh,
                &obj,
                world.materials[obj.material.index]
                    .pbr_metallic_roughness
                    .alpha_texture
                    .as_ref(),
            ) {
                let emissive_factor = obj.material.emissive_factor;
                let some_area_light;
                // only create area light if object material is emissive
                if emissive_factor[0] > 0.0 && emissive_factor[1] > 0.0 && emissive_factor[2] > 0.0
                {
                    let area_light = Arc::new(DiffuseAreaLight::new());
                    lights.push(Arc::clone(&area_light) as Arc<dyn SyncLight>);
                    some_area_light = Some(Arc::clone(&area_light));
                } else {
                    some_area_light = None;
                }

                primitives.push(Arc::new(primitive::GeometricPrimitive::new(
                    shape,
                    Arc::clone(&materials[obj.material.index]),
                    some_area_light,
                )) as Arc<dyn SyncPrimitive>)
            }
        }

        let bvh = Box::new(accelerator::BVH::new(log, primitives, &4)) as Box<dyn SyncPrimitive>;
        let world_bound = bvh.world_bound();

        for light_info in &world.lights {
            lights.push(Light::from_gltf(light_info, &world_bound));
        }

        if lights.is_empty() {
            let default_direction_light = Arc::new(DirectionalLight::new(
                na::convert(na::Translation3::new(1.0, 3.5, 0.0)),
                Spectrum::new(10.0),
                na::Vector3::new(0.0, 1.0, 0.5),
                &world_bound,
            ));
            lights.push(default_direction_light as Arc<dyn SyncLight>);
            let default_point_light = Arc::new(PointLight::new(
                na::convert(na::Translation3::new(1.0, 3.5, 0.0)),
                Spectrum::new(30.0),
            ));
            lights.push(default_point_light as Arc<dyn SyncLight>);
        }

        RenderScene {
            scene: bvh,
            materials,
            lights,
            world_bound,
        }
    }

    pub fn world_bound(&self) -> Bounds3 {
        self.world_bound
    }

    pub fn intersect<'a>(&'a self, r: &Ray, isect: &mut SurfaceInteraction<'a>) -> bool {
        self.scene.intersect(r, isect)
    }

    pub fn intersect_p(&self, r: &Ray) -> bool {
        self.scene.intersect_p(r)
    }
}
