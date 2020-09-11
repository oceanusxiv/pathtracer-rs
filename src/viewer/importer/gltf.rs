use crate::{
    common::importer::gltf::trans_from_gltf,
    viewer::{Mesh, ViewerScene},
};
use std::collections::HashMap;

fn mesh_from_gltf(gltf_prim: &gltf::Primitive, buffers: &[gltf::buffer::Data]) -> Mesh {
    let prim_pos_accessor_idx = gltf_prim.get(&gltf::Semantic::Positions).unwrap().index();

    let reader = gltf_prim.reader(|buffer| Some(&buffers[buffer.index()]));
    Mesh {
        id: prim_pos_accessor_idx,
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
        instances: vec![],
    }
}

fn populate_scene(
    parent_transform: &na::Projective3<f32>,
    current_node: &gltf::Node,
    buffers: &[gltf::buffer::Data],
    mut meshes: &mut Vec<Mesh>,
    mut mesh_prim_indice_map: &mut HashMap<usize, usize>,
) {
    let current_transform = *parent_transform * trans_from_gltf(current_node.transform());
    if let Some(gltf_mesh) = current_node.mesh() {
        for gltf_prim in gltf_mesh.primitives() {
            let prim_pos_accessor_idx = gltf_prim.get(&gltf::Semantic::Positions).unwrap().index();

            if !mesh_prim_indice_map.contains_key(&prim_pos_accessor_idx) {
                mesh_prim_indice_map.insert(prim_pos_accessor_idx, meshes.len());
                meshes.push(mesh_from_gltf(&gltf_prim, buffers));
            }
            let mesh = &mut meshes[mesh_prim_indice_map[&prim_pos_accessor_idx]];
            mesh.instances.push(current_transform);
        }
    }

    for child in current_node.children() {
        populate_scene(
            &current_transform,
            &child,
            &buffers,
            &mut meshes,
            &mut mesh_prim_indice_map,
        );
    }
}

impl ViewerScene {
    pub fn from_gltf(
        document: &gltf::Document,
        buffers: &[gltf::buffer::Data],
        _images: &[gltf::image::Data],
    ) -> Self {
        let mut meshes = vec![];
        let mut mesh_prim_indice_map = HashMap::new();

        for scene in document.scenes() {
            for node in scene.nodes() {
                populate_scene(
                    &na::Projective3::identity(),
                    &node,
                    &buffers,
                    &mut meshes,
                    &mut mesh_prim_indice_map,
                );
            }
        }

        Self { meshes }
    }
}
