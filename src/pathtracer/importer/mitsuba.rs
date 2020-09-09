use crate::{
    common::{importer::mitsuba, spectrum::Spectrum},
    pathtracer::{
        accelerator,
        light::{DiffuseAreaLight, SyncLight},
        material::{Material, MatteMaterial},
        primitive::{GeometricPrimitive, SyncPrimitive},
        shape::{shapes_from_mesh, TriangleMesh},
        texture::{ConstantTexture, SyncTexture},
        RenderScene,
    },
};
use std::{collections::HashMap, sync::Arc};

fn material_from_bsdf(log: &slog::Logger, bsdf: &mitsuba::BSDF) -> Material {
    match bsdf {
        mitsuba::BSDF::TwoSided(bsdf) => material_from_bsdf(&log, &bsdf.bsdf),
        mitsuba::BSDF::Diffuse(bsdf) => Material::Matte(MatteMaterial::new(
            &log,
            Box::new(ConstantTexture::<Spectrum>::new(Spectrum {
                r: bsdf.rgb[0],
                g: bsdf.rgb[1],
                b: bsdf.rgb[2],
            })),
            None,
        )),
    }
}

fn parse_shape(
    shape: &mitsuba::Shape,
    materials: &HashMap<String, Arc<Material>>,
    primitives: &mut Vec<Arc<dyn SyncPrimitive>>,
    lights: &mut Vec<Arc<dyn SyncLight>>,
) {
    let obj_to_world;
    let world_mesh;
    let light_info;
    let material_id;
    match shape {
        mitsuba::Shape::Rectangle {
            transform,
            material,
            emitter,
        } => {
            let mesh = mitsuba::gen_rectangle();
            obj_to_world = transform;
            light_info = emitter;
            material_id = material.id.clone();
            world_mesh = TriangleMesh {
                indices: mesh.indices,
                pos: mesh.pos,
                normal: mesh.normal,
                s: vec![],
                uv: vec![],
                colors: vec![],
                alpha_mask: None,
            };
        }
        mitsuba::Shape::Cube {
            transform,
            material,
            emitter,
        } => {
            let mesh = mitsuba::gen_cube();
            obj_to_world = transform;
            light_info = emitter;
            material_id = material.id.clone();
            world_mesh = TriangleMesh {
                indices: mesh.indices,
                pos: mesh.pos,
                normal: mesh.normal,
                s: vec![],
                uv: vec![],
                colors: vec![],
                alpha_mask: None,
            };
        }
    }

    for shape in shapes_from_mesh(world_mesh, &obj_to_world) {
        let area_light = if let Some(light_info) = light_info {
            if let mitsuba::Emitter::Area { rgb } = light_info {
                let ke = Arc::new(ConstantTexture::<Spectrum>::new(Spectrum {
                    r: rgb[0],
                    g: rgb[1],
                    b: rgb[2],
                })) as Arc<dyn SyncTexture<Spectrum>>;
                let light = Arc::new(DiffuseAreaLight::new(ke, Arc::clone(&shape), 1));
                lights.push(Arc::clone(&light) as Arc<dyn SyncLight>);
                Some(light)
            } else {
                None
            }
        } else {
            None
        };

        primitives.push(Arc::new(GeometricPrimitive::new(
            Arc::clone(&shape),
            Arc::clone(&materials[&material_id]),
            area_light,
        )) as Arc<dyn SyncPrimitive>);
    }
}

impl RenderScene {
    pub fn from_mitsuba(log: &slog::Logger, scene: &mitsuba::Scene) -> Self {
        let mut materials = HashMap::new();
        let mut primitives: Vec<Arc<dyn SyncPrimitive>> = Vec::new();
        let mut lights: Vec<Arc<dyn SyncLight>> = Vec::new();

        for (id, bsdf) in &scene.bsdfs {
            materials.insert(id.clone(), Arc::new(material_from_bsdf(&log, &bsdf)));
        }

        for shape in &scene.shapes {
            parse_shape(&shape, &materials, &mut primitives, &mut lights);
        }

        let bvh = Box::new(accelerator::BVH::new(log, primitives, &4)) as Box<dyn SyncPrimitive>;

        Self { scene: bvh, lights }
    }
}