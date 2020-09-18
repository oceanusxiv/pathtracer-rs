use crate::{
    common::WrapMode,
    common::{importer::mitsuba, spectrum::Spectrum},
    pathtracer::light::InfiniteAreaLight,
    pathtracer::light::Light,
    pathtracer::material::GlassMaterial,
    pathtracer::texture::ImageTexture,
    pathtracer::texture::UVMap,
    pathtracer::{
        accelerator,
        light::{DiffuseAreaLight, SyncLight},
        material::{
            metal::MetalMaterial, schlick_r0_from_eta, substrate::SubstrateMaterial, Material,
            MatteMaterial, MirrorMaterial,
        },
        primitive::{GeometricPrimitive, SyncPrimitive},
        shape::{shapes_from_mesh, TriangleMesh},
        texture::{CheckerTexture, ConstantTexture, SyncTexture},
        Primitive, RenderScene,
    },
};
use std::{collections::HashMap, sync::Arc};

fn texture_from_mitsuba(
    log: &slog::Logger,
    scene_path: &str,
    texture: &mitsuba::Texture,
) -> Box<dyn SyncTexture<Spectrum>> {
    match texture {
        mitsuba::Texture::Checkerboard {
            rgb_params,
            float_params,
        } => Box::new(CheckerTexture::new(
            log,
            Spectrum::from_slice_3(&rgb_params["color0"], false),
            Spectrum::from_slice_3(&rgb_params["color1"], false),
            UVMap::new(
                float_params["uscale"],
                float_params["vscale"],
                float_params["uoffset"],
                float_params["voffset"],
            ),
        )),
        mitsuba::Texture::BitMap { string_params } => {
            let file_path = std::path::Path::new(scene_path)
                .parent()
                .unwrap_or_else(|| std::path::Path::new(""))
                .join(&string_params["filename"]);
            let file_path = file_path.to_str().unwrap();
            let image = image::open(file_path).unwrap();
            match image {
                image::DynamicImage::ImageRgb8(image) => Box::new(ImageTexture::<Spectrum>::new(
                    log,
                    &image,
                    Spectrum::new(1.),
                    WrapMode::Repeat,
                    UVMap::new(1., 1., 0., 0.),
                    true,
                )),
                _ => {
                    panic!("unsupported image format for texture");
                }
            }
        }
    }
}

fn material_from_bsdf(log: &slog::Logger, scene_path: &str, bsdf: &mitsuba::BSDF) -> Material {
    match bsdf {
        mitsuba::BSDF::TwoSided(bsdf) => material_from_bsdf(&log, scene_path, &bsdf.bsdf),
        mitsuba::BSDF::Diffuse(bsdf) => Material::Matte(MatteMaterial::new(
            &log,
            if let Some(texture) = bsdf.texture.as_ref() {
                texture_from_mitsuba(log, scene_path, texture)
            } else {
                Box::new(ConstantTexture::new(Spectrum::from_slice_3(
                    &bsdf.rgb, false,
                )))
            },
        )),
        mitsuba::BSDF::Conductor(bsdf) => {
            if let Some(material) = bsdf.material.as_ref() {
                if material.value == "none" {
                    Material::Mirror(MirrorMaterial::new(log))
                } else {
                    panic!("other material values not supported yet!");
                }
            } else {
                Material::Metal(MetalMaterial::new(
                    &log,
                    Box::new(ConstantTexture::new(Spectrum::from_slice_3(
                        &bsdf.rgb_params["eta"],
                        false,
                    ))),
                    Box::new(ConstantTexture::new(Spectrum::from_slice_3(
                        &bsdf.rgb_params["k"],
                        false,
                    ))),
                    Some(Box::new(ConstantTexture::new(0.001))),
                    None,
                    None,
                    false,
                ))
            }
        }
        mitsuba::BSDF::RoughConductor(bsdf) => Material::Metal(MetalMaterial::new(
            &log,
            Box::new(ConstantTexture::new(Spectrum::from_slice_3(
                &bsdf.rgb_params["eta"],
                false,
            ))),
            Box::new(ConstantTexture::new(Spectrum::from_slice_3(
                &bsdf.rgb_params["k"],
                false,
            ))),
            Some(Box::new(ConstantTexture::new(bsdf.float_params["alpha"]))),
            None,
            None,
            false,
        )),
        mitsuba::BSDF::Dielectric(bsdf) => Material::Glass(GlassMaterial::new(
            &log,
            Box::new(ConstantTexture::new(Spectrum::new(1.0))),
            Box::new(ConstantTexture::new(Spectrum::new(1.0))),
            Box::new(ConstantTexture::new(bsdf.float_params["int_ior"])),
        )),
        // plastic -> substrate is not a perfect match
        mitsuba::BSDF::Plastic(bsdf) => Material::Substrate(SubstrateMaterial::new(
            log,
            if let Some(texture) = bsdf.texture.as_ref() {
                texture_from_mitsuba(log, scene_path, texture)
            } else {
                Box::new(ConstantTexture::new(Spectrum::from_slice_3(
                    &bsdf.rgb_params["diffuse_reflectance"],
                    false,
                )))
            },
            Box::new(ConstantTexture::new(Spectrum::new(schlick_r0_from_eta(
                bsdf.float_params["int_ior"],
            )))),
            Box::new(ConstantTexture::new(0.001)),
            Box::new(ConstantTexture::new(0.001)),
            false,
        )),
        mitsuba::BSDF::RoughPlastic(bsdf) => Material::Substrate(SubstrateMaterial::new(
            log,
            if let Some(texture) = bsdf.texture.as_ref() {
                texture_from_mitsuba(log, scene_path, texture)
            } else {
                Box::new(ConstantTexture::new(Spectrum::from_slice_3(
                    &bsdf.rgb_params["diffuse_reflectance"],
                    false,
                )))
            },
            Box::new(ConstantTexture::new(Spectrum::new(schlick_r0_from_eta(
                bsdf.float_params["int_ior"],
            )))),
            Box::new(ConstantTexture::new(bsdf.float_params["alpha"])),
            Box::new(ConstantTexture::new(bsdf.float_params["alpha"])),
            false,
        )),
    }
}

fn parse_shape(
    log: &slog::Logger,
    scene_path: &str,
    shape: &mitsuba::Shape,
    materials: &HashMap<String, Arc<Material>>,
    primitives: &mut Vec<Arc<dyn SyncPrimitive>>,
    lights: &mut Vec<Arc<dyn SyncLight>>,
) {
    let mut obj_to_world = na::Projective3::identity();
    let world_mesh;
    let light_info;
    let material_ref;
    let material_embed;
    match shape {
        mitsuba::Shape::Rectangle {
            transform,
            material,
            bsdf,
            emitter,
        } => {
            let mesh = mitsuba::gen_rectangle();
            obj_to_world = *transform;
            light_info = emitter;
            material_ref = material;
            material_embed = bsdf;
            world_mesh = TriangleMesh::new_with_transform(
                mesh.indices,
                mesh.pos,
                mesh.normal,
                vec![],
                vec![],
                vec![],
                None,
                &obj_to_world,
            );
        }
        mitsuba::Shape::Cube {
            transform,
            material,
            bsdf,
            emitter,
        } => {
            let mesh = mitsuba::gen_cube();
            obj_to_world = *transform;
            light_info = emitter;
            material_ref = material;
            material_embed = bsdf;
            world_mesh = TriangleMesh::new_with_transform(
                mesh.indices,
                mesh.pos,
                mesh.normal,
                vec![],
                vec![],
                vec![],
                None,
                &obj_to_world,
            );
        }
        mitsuba::Shape::Sphere {
            point,
            radius,
            material,
            bsdf,
            emitter,
        } => {
            let mesh = mitsuba::gen_sphere(point, radius.value);
            light_info = emitter;
            material_ref = material;
            material_embed = bsdf;
            world_mesh = TriangleMesh::new_with_transform(
                mesh.indices,
                mesh.pos,
                mesh.normal,
                vec![],
                vec![],
                vec![],
                None,
                &obj_to_world,
            );
        }
        mitsuba::Shape::Obj {
            transform,
            face_normals,
            material,
            bsdf,
            emitter,
            filename,
        } => {
            let mesh = mitsuba::load_obj(scene_path, filename);
            obj_to_world = *transform;
            light_info = emitter;
            material_ref = material;
            material_embed = bsdf;

            if *face_normals {
                warn!(
                    log,
                    "face normals on for obj, vertex normals will be disregarded"
                );
            }

            world_mesh = TriangleMesh::new_with_transform(
                mesh.indices,
                mesh.pos,
                if *face_normals { vec![] } else { mesh.normal },
                vec![],
                vec![],
                vec![],
                None,
                &obj_to_world,
            );
        }
    }

    let material;
    if let Some(material_ref) = material_ref {
        material = Arc::clone(&materials[&material_ref.id]);
    } else if let Some(material_embed) = material_embed {
        material = Arc::new(material_from_bsdf(&log, scene_path, material_embed));
    } else {
        panic!("either ref exists or embedded bsdf exists");
    }

    for shape in shapes_from_mesh(world_mesh, false) {
        let area_light = if let Some(light_info) = light_info {
            if let mitsuba::Emitter::Area { rgb } = light_info {
                let ke = Arc::new(ConstantTexture::<Spectrum>::new(Spectrum::from_slice_3(
                    rgb, false,
                ))) as Arc<dyn SyncTexture<Spectrum>>;
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
            Arc::clone(&material),
            area_light,
        )) as Arc<dyn SyncPrimitive>);
    }
}

impl RenderScene {
    pub fn from_mitsuba(log: &slog::Logger, scene: &mitsuba::Scene) -> Self {
        let log = log.new(o!("module" => "scene"));
        let mut materials = HashMap::new();
        let mut primitives: Vec<Arc<dyn SyncPrimitive>> = Vec::new();
        let mut lights: Vec<Arc<dyn SyncLight>> = Vec::new();
        let mut infinite_lights: Vec<Arc<dyn SyncLight>> = Vec::new();

        for (id, bsdf) in &scene.bsdfs {
            materials.insert(
                id.clone(),
                Arc::new(material_from_bsdf(&log, &scene.path, &bsdf)),
            );
        }

        for shape in &scene.shapes {
            parse_shape(
                &log,
                &scene.path,
                &shape,
                &materials,
                &mut primitives,
                &mut lights,
            );
        }

        let bvh = Box::new(accelerator::BVH::new(&log, primitives, &4));
        let world_bound = bvh.world_bound();

        // FIXME: should probably figure out what's wrong with the overall transformation
        let env_light_to_world = na::Projective3::from_matrix_unchecked(
            na::Matrix4::from_euler_angles(
                -std::f32::consts::FRAC_PI_2,
                -std::f32::consts::FRAC_PI_2,
                0.0,
            )
            .append_nonuniform_scaling(&na::Vector3::new(1.0, 1.0, -1.0)),
        );

        for emitter in &scene.emitters {
            match emitter {
                mitsuba::Emitter::Area { rgb: _ } => {
                    error!(log, "area lights should not be standalone!");
                }
                mitsuba::Emitter::Point => {} // TODO: support mitsuba point light
                mitsuba::Emitter::EnvMap {
                    transform,
                    filename,
                } => {
                    let file_path = std::path::Path::new(&scene.path)
                        .parent()
                        .unwrap_or_else(|| std::path::Path::new(""))
                        .join(filename);
                    let file_path = file_path.to_str().unwrap();
                    let mut env_light = InfiniteAreaLight::new(
                        &log,
                        transform * env_light_to_world,
                        Spectrum::new(1.0),
                        file_path,
                    );
                    env_light.preprocess(&world_bound);
                    let env_light = Arc::new(env_light) as Arc<dyn SyncLight>;
                    lights.push(Arc::clone(&env_light));
                    infinite_lights.push(Arc::clone(&env_light));
                }
                mitsuba::Emitter::SunSky => {
                    warn!(
                        log,
                        "sunsky emitter not supported, putting default env map instead"
                    );
                    let hdr_map_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                        .join("data/abandoned_tank_farm_04_1k.hdr");
                    let hdr_map_path = hdr_map_path.to_str().unwrap();
                    let mut env_light = InfiniteAreaLight::new(
                        &log,
                        env_light_to_world,
                        Spectrum::new(1.0),
                        hdr_map_path,
                    );
                    env_light.preprocess(&world_bound);
                    let env_light = Arc::new(env_light) as Arc<dyn SyncLight>;
                    lights.push(Arc::clone(&env_light));
                    infinite_lights.push(Arc::clone(&env_light));
                } // TODO: support mitsuba sunsky emitter
            }
        }

        Self {
            scene: bvh,
            lights,
            infinite_lights,
        }
    }
}
