use crate::common::importer::mitsuba;
use crate::viewer::{Mesh, ViewerScene};

impl ViewerScene {
    pub fn from_mitsuba(scene: &mitsuba::Scene) -> Self {
        let mut meshes = vec![];

        for shape in &scene.shapes {
            match shape {
                mitsuba::Shape::Rectangle {
                    transform,
                    material: _,
                    bsdf: _,
                    emitter: _,
                } => {
                    let generated_mesh = mitsuba::gen_rectangle();
                    meshes.push(Mesh {
                        id: 0,
                        indices: generated_mesh.indices,
                        pos: generated_mesh.pos,
                        normal: generated_mesh.normal,
                        s: vec![],
                        uv: vec![],
                        colors: vec![],
                        instances: vec![*transform],
                    })
                }
                mitsuba::Shape::Cube {
                    transform,
                    material: _,
                    bsdf: _,
                    emitter: _,
                } => {
                    let generated_mesh = mitsuba::gen_cube();
                    meshes.push(Mesh {
                        id: 0,
                        indices: generated_mesh.indices,
                        pos: generated_mesh.pos,
                        normal: generated_mesh.normal,
                        s: vec![],
                        uv: vec![],
                        colors: vec![],
                        instances: vec![*transform],
                    })
                }
                mitsuba::Shape::Sphere {
                    point,
                    radius,
                    material: _,
                    bsdf: _,
                    emitter: _,
                } => {
                    let generated_mesh = mitsuba::gen_sphere(&point, radius.value);
                    meshes.push(Mesh {
                        id: 0,
                        indices: generated_mesh.indices,
                        pos: generated_mesh.pos,
                        normal: generated_mesh.normal,
                        s: vec![],
                        uv: vec![],
                        colors: vec![],
                        instances: vec![na::Projective3::identity()],
                    })
                }
                mitsuba::Shape::Obj {
                    transform,
                    face_normals: _,
                    material: _,
                    bsdf: _,
                    emitter: _,
                    filename,
                } => {
                    let obj_mesh = mitsuba::load_obj(&scene.path, filename);
                    meshes.push(Mesh {
                        id: 0,
                        indices: obj_mesh.indices,
                        pos: obj_mesh.pos,
                        normal: obj_mesh.normal,
                        s: vec![],
                        uv: vec![],
                        colors: vec![],
                        instances: vec![*transform],
                    })
                }
            }
        }

        Self { meshes }
    }
}
