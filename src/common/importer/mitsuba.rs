use crate::common::Camera;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;

use serde_xml_rs::from_reader;

pub struct Mesh {
    pub indices: Vec<u32>,
    pub pos: Vec<na::Point3<f32>>,
    pub normal: Vec<na::Vector3<f32>>,
}

pub fn gen_rectangle() -> Mesh {
    Mesh {
        indices: vec![0, 2, 1, 2, 0, 3],
        pos: vec![
            na::Point3::new(-1.0, -1.0, 0.0),
            na::Point3::new(1.0, -1.0, 0.0),
            na::Point3::new(1.0, 1.0, 0.0),
            na::Point3::new(-1.0, 1.0, 0.0),
        ],
        normal: vec![
            glm::vec3(0.0, 0.0, 1.0),
            glm::vec3(0.0, 0.0, 1.0),
            glm::vec3(0.0, 0.0, 1.0),
            glm::vec3(0.0, 0.0, 1.0),
        ],
    }
}

pub fn gen_cube() -> Mesh {
    Mesh {
        indices: vec![
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
        ],
        pos: vec![
            na::Point3::new(-1.0, -1.0, -1.0),
            na::Point3::new(-1.0, -1.0, 1.0),
            na::Point3::new(-1.0, 1.0, 1.0),
            na::Point3::new(1.0, 1.0, -1.0),
            na::Point3::new(-1.0, -1.0, -1.0),
            na::Point3::new(-1.0, 1.0, -1.0),
            na::Point3::new(1.0, -1.0, 1.0),
            na::Point3::new(-1.0, -1.0, -1.0),
            na::Point3::new(1.0, -1.0, -1.0),
            na::Point3::new(1.0, 1.0, -1.0),
            na::Point3::new(1.0, -1.0, -1.0),
            na::Point3::new(-1.0, -1.0, -1.0),
            na::Point3::new(-1.0, -1.0, -1.0),
            na::Point3::new(-1.0, 1.0, 1.0),
            na::Point3::new(-1.0, 1.0, -1.0),
            na::Point3::new(1.0, -1.0, 1.0),
            na::Point3::new(-1.0, -1.0, 1.0),
            na::Point3::new(-1.0, -1.0, -1.0),
            na::Point3::new(-1.0, 1.0, 1.0),
            na::Point3::new(-1.0, -1.0, 1.0),
            na::Point3::new(1.0, -1.0, 1.0),
            na::Point3::new(1.0, 1.0, 1.0),
            na::Point3::new(1.0, -1.0, -1.0),
            na::Point3::new(1.0, 1.0, -1.0),
            na::Point3::new(1.0, -1.0, -1.0),
            na::Point3::new(1.0, 1.0, 1.0),
            na::Point3::new(1.0, -1.0, 1.0),
            na::Point3::new(1.0, 1.0, 1.0),
            na::Point3::new(1.0, 1.0, -1.0),
            na::Point3::new(-1.0, 1.0, -1.0),
            na::Point3::new(1.0, 1.0, 1.0),
            na::Point3::new(-1.0, 1.0, -1.0),
            na::Point3::new(-1.0, 1.0, 1.0),
            na::Point3::new(1.0, 1.0, 1.0),
            na::Point3::new(-1.0, 1.0, 1.0),
            na::Point3::new(1.0, -1.0, 1.0),
        ],
        normal: vec![
            glm::vec3(-1.0, 0.0, 0.0).normalize(),
            glm::vec3(-1.0, 0.0, 0.0).normalize(),
            glm::vec3(-1.0, 0.0, 0.0).normalize(),
            glm::vec3(0.0, 0.0, -1.0).normalize(),
            glm::vec3(0.0, 0.0, -1.0).normalize(),
            glm::vec3(0.0, 0.0, -1.0).normalize(),
            glm::vec3(0.0, -1.0, 0.0).normalize(),
            glm::vec3(0.0, -1.0, 0.0).normalize(),
            glm::vec3(0.0, -1.0, 0.0).normalize(),
            glm::vec3(0.0, 0.0, -1.0).normalize(),
            glm::vec3(0.0, 0.0, -1.0).normalize(),
            glm::vec3(0.0, 0.0, -1.0).normalize(),
            glm::vec3(-1.0, 0.0, 0.0).normalize(),
            glm::vec3(-1.0, 0.0, 0.0).normalize(),
            glm::vec3(-1.0, 0.0, 0.0).normalize(),
            glm::vec3(0.0, -1.0, 0.0).normalize(),
            glm::vec3(0.0, -1.0, 0.0).normalize(),
            glm::vec3(0.0, -1.0, 0.0).normalize(),
            glm::vec3(0.0, 0.0, 1.0).normalize(),
            glm::vec3(0.0, 0.0, 1.0).normalize(),
            glm::vec3(0.0, 0.0, 1.0).normalize(),
            glm::vec3(1.0, 0.0, 0.0).normalize(),
            glm::vec3(1.0, 0.0, 0.0).normalize(),
            glm::vec3(1.0, 0.0, 0.0).normalize(),
            glm::vec3(1.0, 0.0, 0.0).normalize(),
            glm::vec3(1.0, 0.0, 0.0).normalize(),
            glm::vec3(1.0, 0.0, 0.0).normalize(),
            glm::vec3(0.0, 1.0, 0.0).normalize(),
            glm::vec3(0.0, 1.0, 0.0).normalize(),
            glm::vec3(0.0, 1.0, 0.0).normalize(),
            glm::vec3(0.0, 1.0, 0.0).normalize(),
            glm::vec3(0.0, 1.0, 0.0).normalize(),
            glm::vec3(0.0, 1.0, 0.0).normalize(),
            glm::vec3(0.0, 0.0, 1.0).normalize(),
            glm::vec3(0.0, 0.0, 1.0).normalize(),
            glm::vec3(0.0, 0.0, 1.0).normalize(),
        ],
    }
}

#[derive(Debug, Deserialize)]
pub struct Floats {
    name: String,
    value: f32,
}

mod floats {
    use super::Floats;

    use serde::de::{Deserialize, Deserializer};
    use std::collections::HashMap;

    pub fn deserialize<'de, D>(deserializer: D) -> Result<HashMap<String, f32>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let mut map = HashMap::new();
        for item in Vec::<Floats>::deserialize(deserializer)? {
            map.insert(item.name.clone(), item.value);
        }
        Ok(map)
    }
}

#[derive(Debug, Deserialize)]
pub struct Matrix {
    value: String,
}

#[derive(Debug, Deserialize)]
pub struct Transform {
    name: String,
    matrix: Matrix,
}

mod transform {
    use super::Transform;
    use serde::de::{Deserialize, Deserializer};

    pub fn deserialize<'de, D>(deserializer: D) -> Result<na::Projective3<f32>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let transform = Transform::deserialize(deserializer)?;
        let matrix: Vec<f32> = transform
            .matrix
            .value
            .split(" ")
            .map(|s| s.parse().unwrap())
            .collect();
        let mat = na::Matrix4::from_row_slice(&matrix);
        let transform = na::Projective3::from_matrix_unchecked(mat);
        Ok(transform)
    }
}

#[derive(Debug, Deserialize)]
pub struct TwoSided {
    pub id: Option<String>,
    pub bsdf: Box<BSDF>,
}

#[derive(Debug, Deserialize)]
pub struct Diffuse {
    pub id: Option<String>,

    #[serde(with = "rgb")]
    pub rgb: [f32; 3],
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum BSDF {
    #[serde(rename = "twosided")]
    TwoSided(TwoSided),
    #[serde(rename = "diffuse")]
    Diffuse(Diffuse),
}

mod bsdf {
    use super::BSDF;

    use serde::de::{Deserialize, Deserializer};
    use std::collections::HashMap;

    pub fn deserialize<'de, D>(deserializer: D) -> Result<HashMap<String, BSDF>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let mut map = HashMap::new();
        for item in Vec::<BSDF>::deserialize(deserializer)? {
            match item {
                BSDF::TwoSided(item) => {
                    map.insert(item.id.as_ref().unwrap().clone(), BSDF::TwoSided(item));
                }
                BSDF::Diffuse(item) => {
                    map.insert(item.id.as_ref().unwrap().clone(), BSDF::Diffuse(item));
                }
            }
        }
        Ok(map)
    }
}

#[derive(Debug, Deserialize)]
pub struct Rgb {
    name: String,
    value: String,
}

mod rgb {
    use super::Rgb;
    use serde::de::{Deserialize, Deserializer};

    pub fn deserialize<'de, D>(deserializer: D) -> Result<[f32; 3], D::Error>
    where
        D: Deserializer<'de>,
    {
        let rgb = Rgb::deserialize(deserializer)?;
        let color: Vec<f32> = rgb.value.split(", ").map(|s| s.parse().unwrap()).collect();
        Ok([color[0], color[1], color[2]])
    }
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum Emitter {
    #[serde(rename = "area")]
    Area {
        #[serde(with = "rgb")]
        rgb: [f32; 3],
    },
    Point,
}

#[derive(Debug, Deserialize)]
pub struct Reference {
    pub id: String,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum Shape {
    #[serde(rename = "rectangle")]
    Rectangle {
        #[serde(with = "transform")]
        transform: na::Projective3<f32>,

        #[serde(rename = "ref")]
        material: Reference,

        emitter: Option<Emitter>,
    },
    #[serde(rename = "cube")]
    Cube {
        #[serde(with = "transform")]
        transform: na::Projective3<f32>,

        #[serde(rename = "ref")]
        material: Reference,

        emitter: Option<Emitter>,
    },
}

#[derive(Debug, Deserialize)]
pub struct Sensor {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(rename = "float", with = "floats")]
    pub float_params: HashMap<String, f32>,

    #[serde(with = "transform")]
    pub transform: na::Projective3<f32>,
}

#[derive(Debug, Deserialize)]
pub struct Scene {
    pub version: String,
    pub sensor: Sensor,
    #[serde(rename = "bsdf", with = "bsdf")]
    pub bsdfs: HashMap<String, BSDF>,
    #[serde(rename = "shape")]
    pub shapes: Vec<Shape>,
}

fn get_camera(scene: &Scene, resolution: &na::Vector2<f32>) -> Camera {
    let params = &scene.sensor.float_params;
    let fov = params["fov"].to_radians();
    // right to left hand coordinate conversion
    let rotation = na::Rotation3::new(na::Vector3::new(0.0, -std::f32::consts::PI, 0.0));
    let mut cam_to_world: na::Isometry3<f32> =
        na::try_convert(rotation * scene.sensor.transform).unwrap();
    cam_to_world.translation.z *= -1.0;
    Camera::new(
        &cam_to_world,
        &na::Perspective3::new(
            resolution.x / resolution.y,
            fov * (resolution.y / resolution.x),
            0.01,
            10000.0,
        ),
        &resolution,
    )
}

pub fn from_mitsuba(
    log: &slog::Logger,
    path: &str,
    resolution: &na::Vector2<f32>,
) -> (
    Camera,
    crate::pathtracer::RenderScene,
    crate::viewer::ViewerScene,
) {
    let file = File::open(path).unwrap();
    let file = BufReader::new(file);

    let scene: Scene = from_reader(file).unwrap();

    let camera = get_camera(&scene, &resolution);
    let render_scene = crate::pathtracer::RenderScene::from_mitsuba(&log, &scene);
    let viewer_scene = crate::viewer::ViewerScene::from_mitsuba(&log, &scene);

    (camera, render_scene, viewer_scene)
}