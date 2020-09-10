use crate::common::importer::mitsuba;
use crate::viewer::{Mesh, ViewerScene};
use mitsuba::{gen_cube, gen_rectangle, gen_sphere};

impl ViewerScene {
    pub fn from_mitsuba(log: &slog::Logger, scene: &mitsuba::Scene) -> Self {
        let mut meshes = vec![];

        for shape in &scene.shapes {
            match shape {
                mitsuba::Shape::Rectangle {
                    transform,
                    material,
                    emitter,
                } => {
                    let generated_mesh = gen_rectangle();
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
                    material,
                    emitter,
                } => {
                    let generated_mesh = gen_cube();
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
                    material,
                    emitter,
                } => {
                    let generated_mesh = gen_sphere(&point, radius.value);
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
            }
        }

        Self { meshes }
    }
}
