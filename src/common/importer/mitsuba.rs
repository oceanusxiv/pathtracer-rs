use crate::common::Camera;
use genmesh::generators::IndexedPolygon;
use genmesh::generators::SharedVertex;
use genmesh::Triangulate;
use heck::SnakeCase;
use quick_xml::de::from_reader;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::io::Read;
use wavefront_obj::obj;

pub struct Mesh {
    pub indices: Vec<na::Vector3<u32>>,
    pub pos: Vec<na::Point3<f32>>,
    pub normal: Vec<na::Vector3<f32>>,
    pub uv: Vec<na::Point2<f32>>,
}

pub fn gen_rectangle() -> Mesh {
    let plane = genmesh::generators::Plane::new();
    Mesh {
        indices: plane
            .indexed_polygon_iter()
            .triangulate()
            .map(|tr| na::Vector3::new(tr.x as u32, tr.y as u32, tr.z as u32))
            .collect(),
        pos: plane
            .shared_vertex_iter()
            .map(|v| na::Point3::from(na::Vector3::from(v.pos)))
            .collect(),
        normal: plane
            .shared_vertex_iter()
            .map(|v| na::Vector3::from(v.normal))
            .collect(),
        uv: vec![],
    }
}

pub fn gen_cube() -> Mesh {
    let cube = genmesh::generators::Cube::new();
    Mesh {
        indices: cube
            .indexed_polygon_iter()
            .triangulate()
            .map(|tr| na::Vector3::new(tr.x as u32, tr.y as u32, tr.z as u32))
            .collect(),
        pos: cube
            .shared_vertex_iter()
            .map(|v| na::Point3::from(na::Vector3::from(v.pos)))
            .collect(),
        normal: cube
            .shared_vertex_iter()
            .map(|v| na::Vector3::from(v.normal))
            .collect(),
        uv: vec![],
    }
}

pub fn gen_sphere(center: &na::Point3<f32>, radius: f32) -> Mesh {
    let transform = na::Similarity3::new(center.coords, na::Vector3::zeros(), radius);
    let uv_sphere = genmesh::generators::SphereUv::new(10, 10);
    Mesh {
        indices: uv_sphere
            .indexed_polygon_iter()
            .triangulate()
            .map(|tr| na::Vector3::new(tr.x as u32, tr.y as u32, tr.z as u32))
            .collect(),
        pos: uv_sphere
            .shared_vertex_iter()
            .map(|v| transform.transform_point(&na::Point3::from(na::Vector3::from(v.pos))))
            .collect(),
        normal: uv_sphere
            .shared_vertex_iter()
            .map(|v| na::Vector3::from(v.normal))
            .collect(),
        uv: vec![],
    }
}

pub fn load_obj(scene_path: &str, filename: &str) -> Mesh {
    let file_path = std::path::Path::new(scene_path)
        .parent()
        .unwrap_or_else(|| std::path::Path::new(""))
        .join(filename);
    let file_path = file_path.to_str().unwrap();
    let mut input = String::new();

    // load the data directly into memory; no buffering nor streaming
    {
        let mut file = File::open(file_path).unwrap();
        let _ = file.read_to_string(&mut input);
    }
    let obj_file = obj::parse(input).unwrap();

    if obj_file.objects.len() > 1 {
        panic!("only supporting one object right now!");
    }

    let object = &obj_file.objects[0];

    if object.geometry.len() > 1 {
        panic!("only support one set of geometry per object right now!");
    }

    let geometry = &object.geometry[0];

    let mut indices = Vec::new();
    for shape in &geometry.shapes {
        if let obj::Primitive::Triangle((p0, t0, n0), (p1, t1, n1), (p2, t2, n2)) = shape.primitive
        {
            let n0 = n0.unwrap();
            assert_eq!(p0, n0);
            let n1 = n1.unwrap();
            assert_eq!(p1, n1);
            let n2 = n2.unwrap();
            assert_eq!(p2, n2);
            if let Some(t0) = t0 {
                assert_eq!(p0, t0);
            }
            if let Some(t1) = t1 {
                assert_eq!(p1, t1);
            }
            if let Some(t2) = t2 {
                assert_eq!(p2, t2);
            }
            indices.push(na::Vector3::new(p0 as u32, p1 as u32, p2 as u32));
        } else {
            panic!("only support triangle primitives right now!");
        }
    }

    Mesh {
        indices,
        pos: object
            .vertices
            .iter()
            .map(|v| na::Point3::new(v.x as f32, v.y as f32, v.z as f32))
            .collect(),
        normal: object
            .normals
            .iter()
            .map(|n| na::Vector3::new(n.x as f32, n.y as f32, n.z as f32))
            .collect(),
        uv: object
            .tex_vertices
            .iter()
            .map(|t| na::Point2::new(t.u as f32, t.v as f32))
            .collect(),
    }
}

#[derive(Debug, Deserialize)]
pub struct Float {
    #[serde(rename = "@name")]
    pub name: String,

    #[serde(rename = "@value", with = "float")]
    pub value: f32,
}

fn de_floats<'de, D>(deserializer: D) -> Result<HashMap<String, f32>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    struct ChildVisitor;
    impl<'de> serde::de::Visitor<'de> for ChildVisitor {
        type Value = HashMap<String, f32>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter
                .write_str("Map of children elements - filtering for fields with `float` suffix")
        }

        fn visit_map<M>(self, mut access: M) -> Result<Self::Value, M::Error>
        where
            M: serde::de::MapAccess<'de>,
        {
            let mut hm = HashMap::<String, f32>::new();

            while let Some(key) = access.next_key::<String>()? {
                if key.ends_with("float") {
                    let float = access.next_value::<Float>().unwrap();
                    hm.insert(float.name.to_snake_case(), float.value);
                }
            }

            Ok(hm)
        }
    }

    deserializer.deserialize_any(ChildVisitor {})
}

mod float {
    use serde::de::{Deserialize, Deserializer};

    pub fn deserialize<'de, D>(deserializer: D) -> Result<f32, D::Error>
    where
        D: Deserializer<'de>,
    {
        let float = String::deserialize(deserializer)?;
        Ok(float.parse().unwrap())
    }
}

#[derive(Debug, Deserialize)]
pub struct Integer {
    #[serde(rename = "@name")]
    pub name: String,

    #[serde(rename = "@value", with = "integer")]
    pub value: i32,
}

fn de_integers<'de, D>(deserializer: D) -> Result<HashMap<String, i32>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    struct ChildVisitor;
    impl<'de> serde::de::Visitor<'de> for ChildVisitor {
        type Value = HashMap<String, i32>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter
                .write_str("Map of children elements - filtering for fields with `integer` suffix")
        }

        fn visit_map<M>(self, mut access: M) -> Result<Self::Value, M::Error>
        where
            M: serde::de::MapAccess<'de>,
        {
            let mut hm = HashMap::<String, i32>::new();

            while let Some(key) = access.next_key::<String>()? {
                if key.ends_with("integer") {
                    let integer = access.next_value::<Integer>().unwrap();
                    hm.insert(integer.name.to_snake_case(), integer.value);
                }
            }

            Ok(hm)
        }
    }

    deserializer.deserialize_any(ChildVisitor {})
}

mod integer {
    use serde::de::{Deserialize, Deserializer};

    pub fn deserialize<'de, D>(deserializer: D) -> Result<i32, D::Error>
    where
        D: Deserializer<'de>,
    {
        let integer = String::deserialize(deserializer)?;
        Ok(integer.parse().unwrap())
    }
}

#[derive(Debug, Deserialize)]
pub struct Matrix {
    #[serde(rename = "@value")]
    value: String,
}

#[derive(Debug, Deserialize)]
pub struct Transform {
    #[serde(rename = "@name")]
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
    #[serde(rename = "@id")]
    pub id: Option<String>,
    pub bsdf: Box<BSDF>,
}

fn default_rgb_one() -> [f32; 3] {
    [1.0, 1.0, 1.0]
}

#[derive(Debug, Deserialize)]
pub struct Diffuse {
    #[serde(rename = "@id")]
    pub id: Option<String>,

    #[serde(default = "default_rgb_one", with = "rgb")]
    pub rgb: [f32; 3],

    pub texture: Option<Texture>,
}

fn de_rgbs<'de, D>(deserializer: D) -> Result<HashMap<String, [f32; 3]>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    struct ChildVisitor;
    impl<'de> serde::de::Visitor<'de> for ChildVisitor {
        type Value = HashMap<String, [f32; 3]>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("Map of children elements - filtering for fields with `rgb` suffix")
        }

        fn visit_map<M>(self, mut access: M) -> Result<Self::Value, M::Error>
        where
            M: serde::de::MapAccess<'de>,
        {
            let mut hm = HashMap::<String, [f32; 3]>::new();

            while let Some(key) = access.next_key::<String>()? {
                if key.ends_with("rgb") {
                    let rgb = access.next_value::<Rgb>().unwrap();
                    let color: Vec<f32> =
                        rgb.value.split(", ").map(|s| s.parse().unwrap()).collect();
                    hm.insert(rgb.name.to_snake_case(), [color[0], color[1], color[2]]);
                }
            }

            Ok(hm)
        }
    }

    deserializer.deserialize_any(ChildVisitor {})
}

#[derive(Debug, Deserialize)]
pub struct Material {
    #[serde(rename = "@id")]
    pub id: Option<String>,

    #[serde(flatten, rename = "rgb", deserialize_with = "de_rgbs")]
    pub rgb_params: HashMap<String, [f32; 3]>,

    #[serde(flatten, rename = "float", deserialize_with = "de_floats")]
    pub float_params: HashMap<String, f32>,

    #[serde(rename = "string")]
    pub material: Option<StringParam>,

    pub texture: Option<Texture>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "@type")]
pub enum BSDF {
    #[serde(rename = "twosided")]
    TwoSided(TwoSided),
    #[serde(rename = "diffuse")]
    Diffuse(Diffuse),
    #[serde(rename = "conductor")]
    Conductor(Material),
    #[serde(rename = "roughconductor")]
    RoughConductor(Material),
    #[serde(rename = "dielectric")]
    Dielectric(Material),
    #[serde(rename = "plastic")]
    Plastic(Material),
    #[serde(rename = "roughplastic")]
    RoughPlastic(Material),
}

#[macro_export]
macro_rules! bsdf_to_map {
    ($item:expr, $map:expr, $( $x:path ),* ) => {
        match $item {
            $(
                $x(item) => {
                    $map.insert(item.id.as_ref().unwrap().clone(), $x(item));
                }
            )*
        }
    };
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
            bsdf_to_map!(
                item,
                map,
                BSDF::TwoSided,
                BSDF::Diffuse,
                BSDF::Conductor,
                BSDF::RoughConductor,
                BSDF::Dielectric,
                BSDF::Plastic,
                BSDF::RoughPlastic
            )
        }
        Ok(map)
    }
}

#[derive(Debug, Deserialize)]
pub struct Rgb {
    #[serde(rename = "@name")]
    name: String,
    #[serde(rename = "@value")]
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
#[serde(tag = "@type")]
pub enum Emitter {
    #[serde(rename = "area")]
    Area {
        #[serde(with = "rgb")]
        rgb: [f32; 3],
    },
    Point,
    #[serde(rename = "envmap")]
    EnvMap {
        #[serde(with = "transform")]
        transform: na::Projective3<f32>,

        #[serde(rename = "string", with = "string")]
        filename: String,
    },
    #[serde(rename = "sunsky")]
    SunSky,
}

#[derive(Debug, Deserialize)]
pub struct Reference {
    #[serde(rename = "@id")]
    pub id: String,
}

#[derive(Debug, Deserialize)]
pub struct Point {
    pub name: String,
    #[serde(with = "float")]
    pub x: f32,
    #[serde(with = "float")]
    pub y: f32,
    #[serde(with = "float")]
    pub z: f32,
}

mod point {
    use super::Point;
    use serde::de::{Deserialize, Deserializer};

    pub fn deserialize<'de, D>(deserializer: D) -> Result<na::Point3<f32>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let p = Point::deserialize(deserializer)?;
        Ok(na::Point3::new(p.x, p.y, p.z))
    }
}

#[derive(Debug, Deserialize)]
pub struct StringParam {
    pub name: String,
    #[serde(rename = "@value")]
    pub value: String,
}

mod string {
    use super::StringParam;
    use serde::de::{Deserialize, Deserializer};

    pub fn deserialize<'de, D>(deserializer: D) -> Result<String, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = StringParam::deserialize(deserializer)?;
        Ok(s.value)
    }
}

fn de_strings<'de, D>(deserializer: D) -> Result<HashMap<String, String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    struct ChildVisitor;
    impl<'de> serde::de::Visitor<'de> for ChildVisitor {
        type Value = HashMap<String, String>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter
                .write_str("Map of children elements - filtering for fields with `string` suffix")
        }

        fn visit_map<M>(self, mut access: M) -> Result<Self::Value, M::Error>
        where
            M: serde::de::MapAccess<'de>,
        {
            let mut hm = HashMap::<String, String>::new();

            while let Some(key) = access.next_key::<String>()? {
                if key.ends_with("string") {
                    let param = access.next_value::<StringParam>().unwrap();
                    hm.insert(param.name.to_snake_case(), param.value);
                }
            }

            Ok(hm)
        }
    }

    deserializer.deserialize_any(ChildVisitor {})
}

mod bool {
    use super::StringParam;
    use serde::de::{Deserialize, Deserializer};

    pub fn deserialize<'de, D>(deserializer: D) -> Result<bool, D::Error>
    where
        D: Deserializer<'de>,
    {
        let b = StringParam::deserialize(deserializer)?;
        Ok(b.value.parse().unwrap())
    }
}

#[derive(Debug, Deserialize)]
#[serde(tag = "@type")]
pub enum Shape {
    #[serde(rename = "rectangle")]
    Rectangle {
        #[serde(with = "transform")]
        transform: na::Projective3<f32>,

        #[serde(rename = "ref")]
        material: Option<Reference>,

        #[serde(rename = "bsdf")]
        bsdf: Option<BSDF>,

        emitter: Option<Emitter>,
    },
    #[serde(rename = "cube")]
    Cube {
        #[serde(with = "transform")]
        transform: na::Projective3<f32>,

        #[serde(rename = "ref")]
        material: Option<Reference>,

        #[serde(rename = "bsdf")]
        bsdf: Option<BSDF>,

        emitter: Option<Emitter>,
    },
    #[serde(rename = "sphere")]
    Sphere {
        #[serde(with = "point")]
        point: na::Point3<f32>,

        #[serde(rename = "float")]
        radius: Float,

        #[serde(rename = "ref")]
        material: Option<Reference>,

        #[serde(rename = "bsdf")]
        bsdf: Option<BSDF>,

        emitter: Option<Emitter>,
    },
    #[serde(rename = "obj")]
    Obj {
        #[serde(with = "transform")]
        transform: na::Projective3<f32>,
        #[serde(rename = "boolean", default, with = "bool")]
        face_normals: bool,

        #[serde(rename = "ref")]
        material: Option<Reference>,

        #[serde(rename = "bsdf")]
        bsdf: Option<BSDF>,

        emitter: Option<Emitter>,

        #[serde(rename = "string", with = "string")]
        filename: String,
    },
}

#[derive(Debug, Deserialize)]
pub struct Film {
    #[serde(flatten, rename = "integer", deserialize_with = "de_integers")]
    pub integer_params: HashMap<String, i32>,
}

#[derive(Debug, Deserialize)]
pub struct Sensor {
    #[serde(rename = "@type")]
    pub kind: String,

    #[serde(flatten, rename = "float", deserialize_with = "de_floats")]
    pub float_params: HashMap<String, f32>,

    #[serde(with = "transform")]
    pub transform: na::Projective3<f32>,

    pub film: Film,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "@type")]
pub enum Texture {
    #[serde(rename = "checkerboard")]
    Checkerboard {
        #[serde(flatten, rename = "rgb", deserialize_with = "de_rgbs")]
        rgb_params: HashMap<String, [f32; 3]>,

        #[serde(flatten, rename = "float", deserialize_with = "de_floats")]
        float_params: HashMap<String, f32>,
    },
    #[serde(rename = "bitmap")]
    BitMap {
        #[serde(flatten, rename = "string", deserialize_with = "de_strings")]
        string_params: HashMap<String, String>,
    },
}

#[derive(Debug, Deserialize)]
pub struct Scene {
    #[serde(rename = "@version")]
    pub version: String,
    pub sensor: Sensor,
    #[serde(rename = "bsdf", with = "bsdf")]
    pub bsdfs: HashMap<String, BSDF>,
    #[serde(rename = "shape", default)]
    pub shapes: Vec<Shape>,
    #[serde(default, rename = "emitter")]
    pub emitters: Vec<Emitter>,
    #[serde(skip)]
    pub path: String,
}

fn get_camera(scene: &Scene, resolution: &na::Vector2<f32>) -> Camera {
    let params = &scene.sensor.float_params;
    let width = scene.sensor.film.integer_params["width"];
    let height = scene.sensor.film.integer_params["height"];
    let fov = params["fov"].to_radians();
    // right to left hand coordinate conversion
    // FIXME: should probably figure out what's wrong with the overall transformation
    let rotation = na::Rotation3::new(na::Vector3::new(0.0, -std::f32::consts::PI, 0.0));
    // dunno why I need to do this, but sometimes convert to isometry fails even when scaling is 1.0
    let mut sim_cam_to_world: na::Similarity3<f32> =
        na::try_convert(scene.sensor.transform * rotation).unwrap();
    if !(sim_cam_to_world.scaling() == 1.0) {
        sim_cam_to_world.set_scaling(1.0);
    }
    let cam_to_world = sim_cam_to_world.isometry;
    Camera::new(
        &cam_to_world,
        &na::Perspective3::new(
            resolution.x / resolution.y,
            fov * (height as f32 / width as f32),
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
    crate::viewer::renderer::ViewerScene,
) {
    let file = File::open(path).unwrap();
    let file = BufReader::new(file);

    let mut scene: Scene = from_reader(file).unwrap();
    scene.path = String::from(path);

    let camera = get_camera(&scene, &resolution);
    let render_scene = crate::pathtracer::RenderScene::from_mitsuba(&log, &scene);
    let viewer_scene = crate::viewer::renderer::ViewerScene::from_mitsuba(&scene);

    (camera, render_scene, viewer_scene)
}
