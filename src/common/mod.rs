pub mod bounds;
pub mod film;
pub mod math;
pub mod ray;
pub mod spectrum;

use film::Film;
use std::collections::HashMap;
use std::rc::Rc;

lazy_static::lazy_static! {
    pub static ref DEFAULT_RESOLUTION: glm::Vec2 = glm::vec2(640.0, 480.0);
}

static DEFAULT_Z_NEAR: f32 = 0.01;
static DEFAULT_Z_FAR: f32 = 1000.0;

pub struct Camera {
    pub cam_to_world: na::Isometry3<f32>,
    pub cam_to_screen: na::Perspective3<f32>,
    pub screen_to_raster: na::Affine3<f32>,
    pub raster_to_screen: na::Affine3<f32>,

    pub dx_camera: na::Vector3<f32>,
    pub dy_camera: na::Vector3<f32>,

    pub film: Film,
}

impl Camera {
    pub fn new(
        cam_to_world: &na::Isometry3<f32>,
        cam_to_screen: &na::Perspective3<f32>,
        resolution: &glm::Vec2,
    ) -> Camera {
        let screen_to_raster = glm::scaling(&glm::vec3(resolution.x, resolution.y, 1.0))
            * glm::scaling(&glm::vec3(1.0 / (3.0), 1.0 / (-3.0), 1.0))
            * glm::translation(&glm::vec3(1.5, -1.5, 0.0));
        let screen_to_raster = na::Affine3::from_matrix_unchecked(screen_to_raster);
        let resolution = glm::vec2(resolution.x as u32, resolution.y as u32);
        let raster_to_screen = screen_to_raster.inverse();
        let raster_to_camera = cam_to_screen.to_projective().inverse() * raster_to_screen;
        let dx_camera = raster_to_camera * na::Point3::new(1.0, 0.0, 0.0)
            - raster_to_camera * na::Point3::origin();
        let dy_camera = raster_to_camera * na::Point3::new(0.0, 1.0, 0.0)
            - raster_to_camera * na::Point3::origin();

        Camera {
            cam_to_world: *cam_to_world,
            cam_to_screen: *cam_to_screen,
            screen_to_raster,
            raster_to_screen,
            dx_camera,
            dy_camera,
            film: Film::new(&resolution),
        }
    }
}

impl Default for Camera {
    fn default() -> Self {
        Camera::new(
            &na::Isometry3::look_at_rh(
                &na::Point3::new(0.2, 0.05, 0.2),
                &na::Point3::origin(),
                &na::Vector3::new(0.0, 1.0, 0.0),
            )
            .inverse(),
            &na::Perspective3::new(
                DEFAULT_RESOLUTION.x / DEFAULT_RESOLUTION.y,
                std::f32::consts::FRAC_PI_2,
                DEFAULT_Z_NEAR,
                DEFAULT_Z_FAR,
            ),
            &DEFAULT_RESOLUTION,
        )
    }
}

#[derive(Clone)]
pub struct Mesh {
    pub index: usize,
    pub indices: Vec<u32>,
    pub pos: Vec<na::Point3<f32>>,
    pub normal: Vec<na::Vector3<f32>>,
    pub s: Vec<na::Vector3<f32>>,
    pub uv: Vec<na::Point2<f32>>,
    pub colors: Vec<na::Vector3<f32>>,
}

#[derive(Debug)]
pub struct PbrMetallicRoughness {
    pub color_texture: Option<image::RgbImage>,
    pub alpha_texture: Option<image::GrayImage>,
    pub base_color_factor: [f32; 4],
    pub metallic_factor: f32,
    pub roughness_factor: f32,
}
#[derive(Debug)]
pub struct Material {
    pub index: usize,
    pub pbr_metallic_roughness: PbrMetallicRoughness,
}

impl Material {
    pub fn default() -> Self {
        Material {
            index: 0,
            pbr_metallic_roughness: PbrMetallicRoughness {
                color_texture: None,
                alpha_texture: None,
                base_color_factor: [1.0, 1.0, 1.0, 1.0],
                metallic_factor: 1.0,
                roughness_factor: 1.0,
            },
        }
    }
}

pub struct Object {
    pub world_to_obj: na::Projective3<f32>,
    pub obj_to_world: na::Projective3<f32>,
    pub mesh: Rc<Mesh>,
    pub material: Rc<Material>,
}

pub struct World {
    pub objects: Vec<Object>,
    pub meshes: Vec<Rc<Mesh>>,
    pub materials: Vec<Rc<Material>>,

    mesh_prim_indice_map: HashMap<usize, usize>,
}

impl World {
    fn empty() -> World {
        World {
            objects: vec![],
            meshes: vec![],
            materials: vec![Rc::new(Material::default())], // must always have a default material
            mesh_prim_indice_map: HashMap::new(),
        }
    }

    pub fn from_gltf(path: &str) -> (World, Camera) {
        let (document, buffers, images) = gltf::import(path).unwrap();

        let mut camera = Default::default();
        let mut world = World::empty();
        world.populate_materials(&document, &images);
        world.populate_meshes(&document, &buffers);
        for scene in document.scenes() {
            for node in scene.nodes() {
                world.populate_scene(&na::Projective3::identity(), &node);

                if let Some(curr_cam) = World::get_camera(&na::Transform3::identity(), &node) {
                    camera = curr_cam;
                }
            }
        }

        (world, camera)
    }

    fn populate_materials(&mut self, document: &gltf::Document, images: &[gltf::image::Data]) {
        for material in document.materials() {
            let mut color_texture = None;
            let mut alpha_texture = None;
            if let Some(texture) = material.pbr_metallic_roughness().base_color_texture() {
                let image = &images[texture.texture().source().index()];
                match image.format {
                    gltf::image::Format::R8 => {}
                    gltf::image::Format::R8G8 => {}
                    gltf::image::Format::R8G8B8 => {
                        color_texture = image::RgbImage::from_raw(
                            image.width,
                            image.height,
                            image.pixels.clone(),
                        );
                    }
                    gltf::image::Format::R8G8B8A8 => {
                        color_texture = image::RgbImage::from_raw(
                            image.width,
                            image.height,
                            image
                                .pixels
                                .iter()
                                .enumerate()
                                .filter(|&(i, _)| i % 4 != 3)
                                .map(|(_, v)| *v)
                                .collect(),
                        );
                        alpha_texture = image::GrayImage::from_raw(
                            image.width,
                            image.height,
                            image.pixels.iter().skip(3).step_by(4).map(|v| *v).collect(),
                        );
                    }
                    gltf::image::Format::B8G8R8 => {}
                    gltf::image::Format::B8G8R8A8 => {}
                }
            }

            self.materials.push(Rc::new(Material {
                index: self.materials.len(),
                pbr_metallic_roughness: PbrMetallicRoughness {
                    color_texture,
                    alpha_texture,
                    base_color_factor: material.pbr_metallic_roughness().base_color_factor(),
                    metallic_factor: material.pbr_metallic_roughness().metallic_factor(),
                    roughness_factor: material.pbr_metallic_roughness().roughness_factor(),
                },
            }))
        }
    }

    fn populate_meshes(&mut self, document: &gltf::Document, buffers: &[gltf::buffer::Data]) {
        for mesh in document.meshes() {
            for prim in mesh.primitives() {
                let prim_indices_accessor_idx = prim.indices().unwrap().index();
                if !self
                    .mesh_prim_indice_map
                    .contains_key(&prim_indices_accessor_idx)
                {
                    self.mesh_prim_indice_map
                        .insert(prim_indices_accessor_idx, self.meshes.len());

                    let reader = prim.reader(|buffer| Some(&buffers[buffer.index()]));
                    self.meshes.push(Rc::new(Mesh {
                        index: self.meshes.len(),
                        indices: reader.read_indices().unwrap().into_u32().collect(),
                        pos: reader
                            .read_positions()
                            .unwrap()
                            .map(|vertex| na::Point3::from_slice(&vertex))
                            .collect(),
                        normal: match reader.read_normals() {
                            Some(normals) => {
                                normals.map(|normal| glm::make_vec3(&normal)).collect()
                            }
                            None => vec![],
                        },
                        s: match reader.read_tangents() {
                            Some(tangents) => {
                                tangents.map(|tangent| glm::make_vec3(&tangent)).collect()
                            }
                            None => vec![],
                        },
                        uv: match reader.read_tex_coords(0) {
                            Some(read_texels) => read_texels
                                .into_f32()
                                .map(|texel| na::Point2::new(texel[0], texel[1]))
                                .collect(),
                            None => vec![],
                        },
                        colors: match reader.read_colors(0) {
                            Some(colors) => colors
                                .into_rgb_f32()
                                .map(|color| glm::make_vec3(&color))
                                .collect(),
                            None => vec![],
                        },
                    }));
                }
            }
        }
    }

    fn get_camera(
        parent_transform: &na::Transform3<f32>,
        current_node: &gltf::Node,
    ) -> Option<Camera> {
        let current_transform = *parent_transform * from_gltf(current_node.transform());
        if let Some(camera) = current_node.camera() {
            if let gltf::camera::Projection::Perspective(projection) = camera.projection() {
                let zfar = if let Some(far) = projection.zfar() {
                    far
                } else {
                    DEFAULT_Z_FAR
                };
                // TODO: maybe incorporate this in the future
                // let aspect_ratio = if let Some(aspect) = projection.aspect_ratio() {
                //     aspect
                // } else {
                //     std::f32::consts::FRAC_PI_2
                // };
                return Some(Camera::new(
                    &na::try_convert(current_transform).unwrap(),
                    &na::Perspective3::new(
                        DEFAULT_RESOLUTION.x / DEFAULT_RESOLUTION.y,
                        projection.yfov(),
                        projection.znear(),
                        zfar,
                    ),
                    &DEFAULT_RESOLUTION,
                ));
            } else {
                for child in current_node.children() {
                    return World::get_camera(&current_transform, &child);
                }

                None
            }
        } else {
            for child in current_node.children() {
                return World::get_camera(&current_transform, &child);
            }

            None
        }
    }

    fn populate_scene(
        &mut self,
        parent_transform: &na::Projective3<f32>,
        current_node: &gltf::Node,
    ) {
        let current_transform = *parent_transform * from_gltf(current_node.transform());
        if let Some(mesh) = current_node.mesh() {
            for prim in mesh.primitives() {
                let prim_indices_accessor_idx = prim.indices().unwrap().index();
                self.objects.push(Object {
                    world_to_obj: current_transform.inverse(),
                    obj_to_world: current_transform,
                    mesh: Rc::clone(
                        &self.meshes[self.mesh_prim_indice_map[&prim_indices_accessor_idx]],
                    ),
                    material: if let Some(idx) = prim.material().index() {
                        Rc::clone(&self.materials[idx + 1]) // default material on first idx
                    } else {
                        Rc::clone(&self.materials[0])
                    },
                });
            }
        }

        for child in current_node.children() {
            self.populate_scene(&current_transform, &child);
        }
    }
}

fn from_gltf(transform: gltf::scene::Transform) -> na::Projective3<f32> {
    let (translation, rotation, scaling) = transform.decomposed();

    let t = glm::translation(&glm::make_vec3(&translation));
    let r = glm::quat_to_mat4(&glm::make_quat(&rotation));
    let s = glm::scaling(&glm::make_vec3(&scaling));

    na::Projective3::from_matrix_unchecked(t * r * s)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cam_with_look_at(eye: &na::Point3<f32>, center: &na::Point3<f32>) -> Camera {
        Camera::new(
            &na::Isometry3::look_at_rh(eye, center, &glm::vec3(0.0, 1.0, 0.0)).inverse(),
            &na::Perspective3::new(
                DEFAULT_RESOLUTION.x / DEFAULT_RESOLUTION.y,
                std::f32::consts::FRAC_PI_2,
                DEFAULT_Z_NEAR,
                DEFAULT_Z_FAR,
            ),
            &DEFAULT_RESOLUTION,
        )
    }

    #[test]
    fn test_camera_wold_to_screen() {
        let test_cam = cam_with_look_at(&na::Point3::new(10.0, 10.0, 10.0), &na::Point3::origin());

        let test_world_space = na::Point3::origin();
        let test_cam_space = test_cam.cam_to_world.inverse() * test_world_space;
        let test_screen_space = test_cam.cam_to_screen.project_point(&test_cam_space);
        approx::assert_relative_eq!(
            test_cam_space,
            na::Point3::new(0.0, 0.0, -glm::length(&glm::vec3(10.0, 10.0, 10.0))),
            epsilon = 0.000_001
        );

        let z = glm::length(&glm::vec3(10.0, 10.0, 10.0));
        let z_screen =
            ((z - DEFAULT_Z_NEAR) * DEFAULT_Z_FAR) / ((DEFAULT_Z_FAR - DEFAULT_Z_NEAR) * z);

        approx::assert_relative_eq!(
            test_screen_space,
            na::Point3::new(0.0, 0.0, z_screen),
            epsilon = 0.000_001
        );
    }

    #[test]
    fn test_camera_screen_to_raster() {
        let test_cam = cam_with_look_at(&na::Point3::origin(), &na::Point3::new(1.0, 0.0, 0.0));

        let test_screen_space1 = na::Point3::new(1.0, 1.0, 0.5);
        let test_raster_space1 = test_cam.screen_to_raster * test_screen_space1;

        approx::assert_relative_eq!(
            test_raster_space1,
            na::Point3::new(DEFAULT_RESOLUTION.x, DEFAULT_RESOLUTION.y, 0.5),
            epsilon = 0.000_001
        );

        let test_screen_space2 = na::Point3::new(-1.0, -1.0, 0.5);
        let test_raster_space2 = test_cam.screen_to_raster * test_screen_space2;

        approx::assert_relative_eq!(
            test_raster_space2,
            na::Point3::new(0.0, 0.0, 0.5),
            epsilon = 0.000_001
        );
    }

    #[test]
    fn test_camera_raster_to_screen() {
        let test_cam = cam_with_look_at(&na::Point3::origin(), &na::Point3::new(1.0, 0.0, 0.0));

        let test_raster_space1 = na::Point3::new(640.0, 360.0, 0.0);
        let test_cam_space1 = test_cam
            .cam_to_screen
            .unproject_point(&(test_cam.raster_to_screen * test_raster_space1));

        approx::assert_relative_eq!(
            test_cam_space1,
            na::Point3::new(0.0, 0.0, -0.1),
            epsilon = 0.000_001
        );
    }
}
