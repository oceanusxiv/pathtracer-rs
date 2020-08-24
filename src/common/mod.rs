use image::RgbImage;
use std::collections::HashMap;
use std::rc::Rc;

lazy_static::lazy_static! {
    pub static ref DEFAULT_RESOLUTION: glm::Vec2 = glm::vec2(1280.0, 720.0);
}

static DEFAULT_Z_NEAR: f32 = 0.1;
static DEFAULT_Z_FAR: f32 = 1000.0;

pub struct Camera {
    pub cam_to_world: glm::Mat4,
    pub cam_to_screen: glm::Mat4,
    pub screen_to_raster: glm::Mat4,
    pub raster_to_screen: glm::Mat4,
    pub raster_to_cam: glm::Mat4,
    pub image: Box<RgbImage>,
}

impl Camera {
    pub fn new(
        cam_to_world: &glm::Mat4,
        cam_to_screen: &glm::Mat4,
        resolution: &glm::Vec2,
    ) -> Camera {
        let screen_to_raster =
            glm::scaling(&glm::vec3(resolution.x * 0.5, resolution.y * 0.5, 1.0))
                * glm::translation(&glm::vec3(1.0, 1.0, 0.0));
        let raster_to_screen = glm::inverse(&screen_to_raster);
        let resolution = glm::vec2(resolution.x as u32, resolution.y as u32);
        Camera {
            cam_to_world: *cam_to_world,
            cam_to_screen: *cam_to_screen,
            screen_to_raster,
            raster_to_screen,
            raster_to_cam: glm::inverse(&cam_to_screen) * raster_to_screen,
            image: Box::new(RgbImage::new(resolution.x, resolution.y)),
        }
    }

    pub fn default() -> Camera {
        Camera::new(
            &glm::inverse(&glm::look_at(
                &glm::vec3(0.2, 0.05, 0.2),
                &glm::vec3(0.0, 0.0, 0.0),
                &glm::vec3(0.0, 1.0, 0.0),
            )),
            &glm::perspective_zo(
                DEFAULT_RESOLUTION.x / DEFAULT_RESOLUTION.y,
                std::f32::consts::FRAC_PI_2,
                DEFAULT_Z_NEAR,
                DEFAULT_Z_FAR,
            ),
            &DEFAULT_RESOLUTION,
        )
    }
}

pub struct Mesh {
    pub index: usize,
    pub indices: Vec<u32>,
    pub pos: Vec<glm::Vec3>,
    pub normal: Vec<glm::Vec3>,
}

pub struct TBounds3<T: glm::Scalar> {
    pub p_min: glm::TVec3<T>,
    pub p_max: glm::TVec3<T>,
}

pub type Bounds3 = TBounds3<f32>;

pub struct Object {
    pub world_to_obj: glm::Mat4,
    pub obj_to_world: glm::Mat4,
    pub mesh: Rc<Mesh>,
}

pub trait Light {}

pub struct World {
    pub objects: Vec<Object>,
    pub meshes: Vec<Rc<Mesh>>,

    mesh_prim_indice_map: HashMap<[usize; 2], usize>,
}

impl World {
    fn empty() -> World {
        World {
            objects: vec![],
            meshes: vec![],
            mesh_prim_indice_map: HashMap::new(),
        }
    }

    pub fn from_gltf(path: &str) -> (World, Camera) {
        let (document, buffers, images) = gltf::import(path).unwrap();

        let mut camera = Camera::default();
        let mut world = World::empty();
        world.populate_meshes(&document, &buffers);
        for scene in document.scenes() {
            for node in scene.nodes() {
                world.populate_scene(&glm::identity(), &node);

                if let Some(curr_cam) = World::get_camera(&glm::identity(), &node) {
                    camera = curr_cam;
                }
            }
        }

        (world, camera)
    }

    fn populate_materials(&mut self, document: &gltf::Document) {
        for material in document.materials() {}
    }

    fn populate_meshes(&mut self, document: &gltf::Document, buffers: &[gltf::buffer::Data]) {
        for mesh in document.meshes() {
            for prim in mesh.primitives() {
                self.mesh_prim_indice_map
                    .insert([mesh.index(), prim.index()], self.meshes.len());

                let reader = prim.reader(|buffer| Some(&buffers[buffer.index()]));
                self.meshes.push(Rc::new(Mesh {
                    index: self.meshes.len(),
                    indices: reader.read_indices().unwrap().into_u32().collect(),
                    pos: reader
                        .read_positions()
                        .unwrap()
                        .map(|vertex| glm::make_vec3(&vertex))
                        .collect(),
                    normal: match reader.read_normals() {
                        Some(normals) => normals.map(|normal| glm::make_vec3(&normal)).collect(),
                        None => vec![],
                    },
                }));
            }
        }
    }

    fn get_camera(parent_transform: &glm::Mat4, current_node: &gltf::Node) -> Option<Camera> {
        let current_transform = *parent_transform * from_gltf(current_node.transform());
        if let Some(camera) = current_node.camera() {
            if let gltf::camera::Projection::Perspective(projection) = camera.projection() {
                let zfar = if let Some(far) = projection.zfar() {
                    far
                } else {
                    DEFAULT_Z_FAR
                };
                let aspect_ratio = if let Some(aspect) = projection.aspect_ratio() {
                    aspect
                } else {
                    std::f32::consts::FRAC_PI_2
                };
                return Some(Camera::new(
                    &current_transform,
                    &glm::perspective_zo(
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

    fn populate_scene(&mut self, parent_transform: &glm::Mat4, current_node: &gltf::Node) {
        let current_transform = *parent_transform * from_gltf(current_node.transform());
        if let Some(mesh) = current_node.mesh() {
            for prim in mesh.primitives() {
                self.objects.push(Object {
                    world_to_obj: glm::inverse(&current_transform),
                    obj_to_world: current_transform,
                    mesh: Rc::clone(
                        &self.meshes[self.mesh_prim_indice_map[&[mesh.index(), prim.index()]]],
                    ),
                });
            }
        }

        for child in current_node.children() {
            self.populate_scene(&current_transform, &child);
        }
    }
}

fn from_gltf(transform: gltf::scene::Transform) -> glm::Mat4 {
    let (translation, rotation, scaling) = transform.decomposed();

    let t = glm::translation(&glm::make_vec3(&translation));
    let r = glm::quat_to_mat4(&glm::make_quat(&rotation));
    let s = glm::scaling(&glm::make_vec3(&scaling));

    t * r * s
}

mod tests {
    use super::*;

    fn cam_with_look_at(eye: &glm::Vec3, center: &glm::Vec3) -> Camera {
        Camera::new(
            &glm::inverse(&glm::look_at(eye, center, &glm::vec3(0.0, 1.0, 0.0))),
            &glm::perspective_zo(
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
        let test_cam = cam_with_look_at(&glm::vec3(10.0, 10.0, 10.0), &glm::vec3(0.0, 0.0, 0.0));

        let test_world_space = glm::vec4(0.0, 0.0, 0.0, 1.0);
        let test_cam_space = glm::inverse(&test_cam.cam_to_world) * test_world_space;
        let test_screen_space = test_cam.cam_to_screen * test_cam_space;
        approx::assert_relative_eq!(
            test_cam_space / test_cam_space.w,
            glm::vec4(0.0, 0.0, -glm::length(&glm::vec3(10.0, 10.0, 10.0)), 1.0),
            epsilon = 0.000_001
        );

        let z = glm::length(&glm::vec3(10.0, 10.0, 10.0));
        let z_screen =
            ((z - DEFAULT_Z_NEAR) * DEFAULT_Z_FAR) / ((DEFAULT_Z_FAR - DEFAULT_Z_NEAR) * z);

        approx::assert_relative_eq!(
            test_screen_space / test_screen_space.w,
            glm::vec4(0.0, 0.0, z_screen, 1.0),
            epsilon = 0.000_001
        );
    }

    #[test]
    fn test_camera_screen_to_raster() {
        let test_cam = cam_with_look_at(&glm::vec3(0.0, 0.0, 0.0), &glm::vec3(1.0, 0.0, 0.0));

        let test_screen_space1 = glm::vec4(1.0, 1.0, 0.5, 1.0);
        let test_raster_space1 = test_cam.screen_to_raster * test_screen_space1;

        approx::assert_relative_eq!(
            test_raster_space1 / test_raster_space1.w,
            glm::vec4(DEFAULT_RESOLUTION.x, DEFAULT_RESOLUTION.y, 0.5, 1.0),
            epsilon = 0.000_001
        );

        let test_screen_space2 = glm::vec4(-1.0, -1.0, 0.5, 1.0);
        let test_raster_space2 = test_cam.screen_to_raster * test_screen_space2;

        approx::assert_relative_eq!(
            test_raster_space2 / test_raster_space2.w,
            glm::vec4(0.0, 0.0, 0.5, 1.0),
            epsilon = 0.000_001
        );
    }

    #[test]
    fn test_camera_raster_to_screen() {
        let test_cam = cam_with_look_at(&glm::vec3(0.0, 0.0, 0.0), &glm::vec3(1.0, 0.0, 0.0));

        let test_raster_space1 = glm::vec4(640.0, 360.0, 0.0, 1.0);
        let test_cam_space1 = test_cam.raster_to_cam * test_raster_space1;

        approx::assert_relative_eq!(
            test_cam_space1 / test_cam_space1.w,
            glm::vec4(0.0, 0.0, -0.1, 1.0),
            epsilon = 0.000_001
        );
    }
}
