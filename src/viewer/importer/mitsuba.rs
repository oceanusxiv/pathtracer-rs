use crate::common::importer::mitsuba;
use crate::viewer::{Mesh, ViewerScene};
use mitsuba::gen_rectangle;

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
                } => {}
            }
        }

        Self { meshes }
    }
}
