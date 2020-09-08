use crate::{
    common::{importer::gltf::trans_from_gltf, spectrum::Spectrum, WrapMode},
    pathtracer::{
        accelerator,
        light::{DiffuseAreaLight, DirectionalLight, PointLight, SyncLight},
        material::{GlassMaterial, Material, MatteMaterial, MirrorMaterial},
        primitive::{GeometricPrimitive, SyncPrimitive},
        shape::{shapes_from_mesh, SyncShape, TriangleMesh},
        texture::{ConstantTexture, ImageTexture, NormalMap, SyncTexture, UVMap},
        RenderScene,
    },
};
use std::sync::Arc;

impl ImageTexture<f32> {}

pub fn default_material(log: &slog::Logger) -> Material {
    let color_factor = Spectrum::new(1.0);
    let color_texture =
        Box::new(ConstantTexture::<Spectrum>::new(color_factor)) as Box<dyn SyncTexture<Spectrum>>;

    Material::Matte(MatteMaterial::new(log, color_texture, None))
}

pub fn material_from_gltf(
    log: &slog::Logger,
    gltf_material: &gltf::Material,
    images: &[gltf::image::Data],
) -> Material {
    let pbr = &gltf_material.pbr_metallic_roughness();
    let color_factor = Spectrum::from_slice(&pbr.base_color_factor(), true);
    let mut color_texture =
        Box::new(ConstantTexture::<Spectrum>::new(color_factor)) as Box<dyn SyncTexture<Spectrum>>;
    let mut normal_map = None;

    if let Some(texture) = pbr.base_color_texture() {
        let image = &images[texture.texture().source().index()];
        let sampler = &texture.texture().sampler();
        assert_eq!(sampler.wrap_s(), sampler.wrap_t());
        let wrap_mode = match sampler.wrap_s() {
            gltf::texture::WrappingMode::ClampToEdge => WrapMode::Clamp,
            gltf::texture::WrappingMode::MirroredRepeat => WrapMode::Repeat,
            gltf::texture::WrappingMode::Repeat => WrapMode::Repeat,
        };

        match image.format {
            gltf::image::Format::R8G8B8 => {
                if let Some(image) =
                    image::RgbImage::from_raw(image.width, image.height, image.pixels.clone())
                {
                    color_texture = Box::new(ImageTexture::<Spectrum>::new(
                        log,
                        &image,
                        color_factor,
                        wrap_mode,
                        UVMap::new(image.width() as f32, image.height() as f32, 0.0, 0.0),
                        true,
                    )) as Box<dyn SyncTexture<Spectrum>>;
                }
            }
            gltf::image::Format::R8G8B8A8 => {
                if let Some(image) = image::RgbImage::from_raw(
                    image.width,
                    image.height,
                    image
                        .pixels
                        .iter()
                        .enumerate()
                        .filter(|&(i, _)| i % 4 != 3)
                        .map(|(_, v)| *v)
                        .collect(),
                ) {
                    color_texture = Box::new(ImageTexture::<Spectrum>::new(
                        log,
                        &image,
                        color_factor,
                        wrap_mode,
                        UVMap::new(image.width() as f32, image.height() as f32, 0.0, 0.0),
                        true,
                    )) as Box<dyn SyncTexture<Spectrum>>;
                }
            }
            _ => error!(
                log,
                "unsupported image format {:?} for color texture", image.format
            ),
        }
    }

    if let Some(texture) = gltf_material.normal_texture().as_ref() {
        let image = &images[texture.texture().source().index()];
        let sampler = &texture.texture().sampler();
        assert_eq!(sampler.wrap_s(), sampler.wrap_t());
        let wrap_mode = match sampler.wrap_s() {
            gltf::texture::WrappingMode::ClampToEdge => WrapMode::Clamp,
            gltf::texture::WrappingMode::MirroredRepeat => WrapMode::Repeat,
            gltf::texture::WrappingMode::Repeat => WrapMode::Repeat,
        };
        let image =
            image::RgbImage::from_raw(image.width, image.height, image.pixels.clone()).unwrap();
        normal_map = Some(Box::new(NormalMap::new(
            log,
            &image,
            na::Vector2::new(texture.scale(), texture.scale()),
            wrap_mode,
            UVMap::new(image.width() as f32, image.height() as f32, 0.0, 0.0),
        )) as Box<dyn SyncTexture<na::Vector3<f32>>>);
    }

    // TODO: get gltf to work with transmission
    let transmission_factor = 0.0;
    let ior = 1.5;
    if transmission_factor > 0.0 {
        let index = Box::new(ConstantTexture::<f32>::new(ior)) as Box<dyn SyncTexture<f32>>;
        let reflect_color = Box::new(ConstantTexture::<Spectrum>::new(Spectrum::new(1.0)))
            as Box<dyn SyncTexture<Spectrum>>;
        let transmit_color = Box::new(ConstantTexture::<Spectrum>::new(Spectrum::new(1.0)))
            as Box<dyn SyncTexture<Spectrum>>;
        return Material::Glass(GlassMaterial::new(
            log,
            reflect_color,
            transmit_color,
            index,
        ));
    }

    if pbr.metallic_factor() == 1.0 && pbr.roughness_factor() == 0.0 {
        return Material::Mirror(MirrorMaterial::new(log));
    }

    if pbr.metallic_factor() == 0.0 && pbr.roughness_factor() == 0.0 {
        return Material::Matte(MatteMaterial::new(log, color_texture, normal_map));
    }

    Material::Matte(MatteMaterial::new(log, color_texture, normal_map))
}

pub fn shapes_from_gltf_prim(
    log: &slog::Logger,
    gltf_prim: &gltf::Primitive,
    obj_to_world: &na::Projective3<f32>,
    images: &[gltf::image::Data],
    buffers: &[gltf::buffer::Data],
) -> Vec<Arc<dyn SyncShape>> {
    let mut alpha_mask_texture = None;

    if let Some(texture) = gltf_prim
        .material()
        .pbr_metallic_roughness()
        .base_color_texture()
    {
        let image = &images[texture.texture().source().index()];
        let sampler = &texture.texture().sampler();
        assert_eq!(sampler.wrap_s(), sampler.wrap_t());
        let wrap_mode = match sampler.wrap_s() {
            gltf::texture::WrappingMode::ClampToEdge => WrapMode::Clamp,
            gltf::texture::WrappingMode::MirroredRepeat => WrapMode::Repeat,
            gltf::texture::WrappingMode::Repeat => WrapMode::Repeat,
        };

        match gltf_prim.material().alpha_mode() {
            gltf::material::AlphaMode::Mask => {
                assert!(image.format == gltf::image::Format::R8G8B8A8);
                if let Some(image) = image::GrayImage::from_raw(
                    image.width,
                    image.height,
                    image.pixels.iter().skip(3).step_by(4).map(|v| *v).collect(),
                ) {
                    alpha_mask_texture = Some(Arc::new(ImageTexture::<f32>::new(
                        log,
                        &image,
                        1.0,
                        wrap_mode,
                        UVMap::new(image.width() as f32, image.height() as f32, 0.0, 0.0),
                    )) as Arc<dyn SyncTexture<f32>>);
                }
            }
            _ => {}
        }
    }

    let reader = gltf_prim.reader(|buffer| Some(&buffers[buffer.index()]));
    let mut world_mesh = TriangleMesh {
        indices: reader.read_indices().unwrap().into_u32().collect(),
        pos: reader
            .read_positions()
            .unwrap()
            .map(|vertex| na::Point3::from_slice(&vertex))
            .collect(),
        normal: match reader.read_normals() {
            Some(normals) => normals.map(|normal| glm::make_vec3(&normal)).collect(),
            None => vec![],
        },
        s: match reader.read_tangents() {
            Some(tangents) => tangents.map(|tangent| glm::make_vec3(&tangent)).collect(),
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
        alpha_mask: alpha_mask_texture,
    };

    shapes_from_mesh(world_mesh, &obj_to_world)
}

fn populate_scene(
    log: &slog::Logger,
    parent_transform: &na::Projective3<f32>,
    current_node: &gltf::Node,
    buffers: &[gltf::buffer::Data],
    images: &[gltf::image::Data],
    materials: &Vec<Arc<Material>>,
    mut primitives: &mut Vec<Arc<dyn SyncPrimitive>>,
    mut lights: &mut Vec<Arc<dyn SyncLight>>,
    mut preprocess_lights: &mut Vec<Arc<dyn SyncLight>>,
) {
    let current_transform = *parent_transform * trans_from_gltf(current_node.transform());
    const EMISSIVE_SCALING_FACTOR: f32 = 10.0; // hack for gltf since it clamps emissive factor to 1.0
    if let Some(gltf_mesh) = current_node.mesh() {
        for gltf_prim in gltf_mesh.primitives() {
            for shape in
                shapes_from_gltf_prim(log, &gltf_prim, &current_transform, &images, buffers)
            {
                let emissive_factor = gltf_prim.material().emissive_factor();
                let some_area_light;
                // only create area light if object material is emissive
                if emissive_factor[0] > 0.0 || emissive_factor[1] > 0.0 || emissive_factor[2] > 0.0
                {
                    let area_light = Arc::new(DiffuseAreaLight::new(
                        Spectrum {
                            r: EMISSIVE_SCALING_FACTOR * emissive_factor[0],
                            g: EMISSIVE_SCALING_FACTOR * emissive_factor[0],
                            b: EMISSIVE_SCALING_FACTOR * emissive_factor[0],
                        },
                        Arc::clone(&shape),
                        1,
                    ));
                    lights.push(Arc::clone(&area_light) as Arc<dyn SyncLight>);
                    some_area_light = Some(Arc::clone(&area_light));
                } else {
                    some_area_light = None;
                }

                primitives.push(Arc::new(GeometricPrimitive::new(
                    shape,
                    if let Some(idx) = gltf_prim.material().index() {
                        Arc::clone(&materials[idx + 1]) // default material on first idx
                    } else {
                        Arc::clone(&materials[0])
                    },
                    some_area_light,
                )) as Arc<dyn SyncPrimitive>)
            }
        }
    }

    if let Some(light) = current_node.light() {
        let light_color = Spectrum {
            r: light.intensity() * light.color()[0],
            g: light.intensity() * light.color()[0],
            b: light.intensity() * light.color()[0],
        };
        match light.kind() {
            gltf::khr_lights_punctual::Kind::Directional => {
                preprocess_lights.push(Arc::new(DirectionalLight::new(
                    &current_transform,
                    light_color,
                    na::Vector3::new(0.0, 0.0, -1.0),
                )));
            }

            gltf::khr_lights_punctual::Kind::Point => {
                lights.push(Arc::new(PointLight::new(&current_transform, light_color)));
            }

            // TODO: implement spotlight
            gltf::khr_lights_punctual::Kind::Spot {
                inner_cone_angle,
                outer_cone_angle,
            } => {
                lights.push(Arc::new(PointLight::new(&current_transform, light_color)));
            }
        }
    }

    for child in current_node.children() {
        populate_scene(
            &log,
            &current_transform,
            &child,
            &buffers,
            &images,
            &materials,
            &mut primitives,
            &mut lights,
            &mut preprocess_lights,
        );
    }
}

impl RenderScene {
    pub fn from_gltf(
        log: &slog::Logger,
        document: &gltf::Document,
        buffers: &[gltf::buffer::Data],
        images: &[gltf::image::Data],
    ) -> Self {
        let mut primitives: Vec<Arc<dyn SyncPrimitive>> = Vec::new();
        let mut materials = vec![Arc::new(default_material(log))];
        let mut lights: Vec<Arc<dyn SyncLight>> = Vec::new();
        let mut preprocess_lights: Vec<Arc<dyn SyncLight>> = Vec::new();

        for material in document.materials() {
            materials.push(Arc::new(material_from_gltf(log, &material, &images)));
        }

        for scene in document.scenes() {
            for node in scene.nodes() {
                populate_scene(
                    log,
                    &na::Projective3::identity(),
                    &node,
                    &buffers,
                    &images,
                    &materials,
                    &mut primitives,
                    &mut lights,
                    &mut preprocess_lights,
                );
            }
        }

        let bvh = Box::new(accelerator::BVH::new(log, primitives, &4)) as Box<dyn SyncPrimitive>;
        let world_bound = bvh.world_bound();

        if lights.is_empty() && preprocess_lights.is_empty() {
            let default_direction_light = Arc::new(DirectionalLight::new(
                &na::convert(na::Translation3::new(1.0, 3.5, 0.0)),
                Spectrum::new(10.0),
                na::Vector3::new(0.0, 1.0, 0.5),
            ));
            preprocess_lights.push(default_direction_light as Arc<dyn SyncLight>);
            let default_point_light = Arc::new(PointLight::new(
                &na::convert(na::Translation3::new(1.0, 3.5, 0.0)),
                Spectrum::new(30.0),
            ));
            lights.push(default_point_light as Arc<dyn SyncLight>);
        }

        // run preprocess for lights that need it
        for mut light in preprocess_lights.into_iter() {
            Arc::get_mut(&mut light).unwrap().preprocess(&world_bound);
            lights.push(Arc::clone(&light));
        }

        Self { scene: bvh, lights }
    }
}
