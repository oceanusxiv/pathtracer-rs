use super::primitive::Primitive;
use super::{SurfaceInteraction, SyncPrimitive};
use crate::common::bounds::{Bounds3, TBounds3};
use crate::common::ray::Ray;
use std::{sync::Arc, time::Instant};

struct BVHPrimitiveInfo {
    pub prim_num: usize,
    pub centroid: na::Point3<f32>,
    pub bounds: Bounds3,
}

impl BVHPrimitiveInfo {
    fn new(prim_num: usize, bounds: Bounds3) -> Self {
        BVHPrimitiveInfo {
            prim_num,
            centroid: bounds.p_min + 0.5 * (bounds.p_max - bounds.p_min),
            bounds,
        }
    }
}

struct BVHBuildNode {
    bounds: Bounds3,
    children: [Option<Box<BVHBuildNode>>; 2],
    split_axis: usize,
    first_prim_offset: usize,
    num_prims: usize,
}

impl BVHBuildNode {
    pub fn new_leaf(first: usize, n: usize, b: Bounds3) -> Self {
        BVHBuildNode {
            bounds: b,
            children: [None, None],
            split_axis: 0,
            first_prim_offset: first,
            num_prims: n,
        }
    }

    pub fn new_interior(axis: usize, c0: Box<BVHBuildNode>, c1: Box<BVHBuildNode>) -> Self {
        BVHBuildNode {
            bounds: Bounds3::union(&c0.bounds, &c1.bounds),
            children: [Some(c0), Some(c1)],
            split_axis: axis,
            first_prim_offset: 0,
            num_prims: 0,
        }
    }
}

#[derive(Copy, Clone)]
struct BucketInfo {
    count: usize,
    bounds: Bounds3,
}

impl BucketInfo {
    fn new() -> Self {
        BucketInfo {
            count: 0,
            bounds: Bounds3::empty(),
        }
    }
}

#[repr(C)]
union LinearBVHOffset {
    primitives_offset: u32,
    second_child_offset: u32,
}

#[repr(C, align(32))]
struct LinearBVHNode {
    bounds: Bounds3,
    offset: LinearBVHOffset,
    num_prims: u16,
    axis: u8,
}

pub struct BVH {
    primitives: Vec<Arc<dyn SyncPrimitive>>,
    nodes: Box<[LinearBVHNode]>,
}

impl BVH {
    pub fn new(primitives: Vec<Arc<dyn SyncPrimitive>>, max_prims_in_node: &usize) -> Self {
        let start = Instant::now();

        let mut primitive_info = Vec::<BVHPrimitiveInfo>::with_capacity(primitives.len());

        for i in 0..primitives.len() {
            primitive_info.push(BVHPrimitiveInfo::new(i, primitives[i].world_bound()))
        }

        let mut total_nodes = 0usize;
        let mut ordered_prims = Vec::<Arc<dyn SyncPrimitive>>::with_capacity(primitives.len());

        let root = BVH::recursive_build(
            &mut primitive_info,
            max_prims_in_node,
            0,
            primitives.len(),
            &mut total_nodes,
            &mut ordered_prims,
            &primitives,
        );

        let mut nodes = Box::<[LinearBVHNode]>::new_uninit_slice(total_nodes);
        let mut offset = 0usize;
        BVH::flatten_bvh_tree(&root, &mut nodes, &mut offset);

        let duration = start.elapsed();
        debug!("bvh tree took {:?} to construct", duration);
        let nodes = unsafe { nodes.assume_init() };
        BVH {
            primitives: ordered_prims,
            nodes,
        }
    }

    fn recursive_build(
        primitive_info: &mut Vec<BVHPrimitiveInfo>,
        max_prims_in_node: &usize,
        start: usize,
        end: usize,
        total_size: &mut usize,
        ordered_prims: &mut Vec<Arc<dyn SyncPrimitive>>,
        primitives: &Vec<Arc<dyn SyncPrimitive>>,
    ) -> Box<BVHBuildNode> {
        *total_size += 1;

        let mut bounds = Bounds3::empty();
        for i in start..end {
            bounds = Bounds3::union(&bounds, &primitive_info[i].bounds);
        }
        let num_prims = end - start;

        if num_prims == 1 {
            let first_prim_offset = ordered_prims.len();
            for i in start..end {
                let prim_num = primitive_info[i].prim_num;
                ordered_prims.push(Arc::clone(&primitives[prim_num]));
            }

            return Box::new(BVHBuildNode::new_leaf(first_prim_offset, num_prims, bounds));
        } else {
            let mut centroid_bounds = Bounds3::empty();

            for i in start..end {
                centroid_bounds = Bounds3::union_p(&centroid_bounds, &primitive_info[i].centroid);
            }

            let dim = centroid_bounds.maximum_extent();
            let mut mid = (start + end) / 2;
            if centroid_bounds.p_max[dim] == centroid_bounds.p_min[dim] {
                let first_prim_offset = ordered_prims.len();
                for i in start..end {
                    let prim_num = primitive_info[i].prim_num;
                    ordered_prims.push(Arc::clone(&primitives[prim_num]));
                }

                return Box::new(BVHBuildNode::new_leaf(first_prim_offset, num_prims, bounds));
            } else {
                if num_prims <= 2 {
                    mid = (start + end) / 2;

                    primitive_info[start..end].partition_at_index_by(mid - start, |a, b| {
                        a.centroid[dim].partial_cmp(&b.centroid[dim]).unwrap()
                    });
                } else {
                    const N_BUCKETS: usize = 12;
                    let mut buckets = [BucketInfo::new(); N_BUCKETS];

                    for i in start..end {
                        let mut b = (N_BUCKETS as f32
                            * centroid_bounds.offset(&primitive_info[i].centroid)[dim])
                            as usize;
                        if b == N_BUCKETS {
                            b = N_BUCKETS - 1;
                        }

                        buckets[b].count += 1;
                        buckets[b].bounds =
                            Bounds3::union(&buckets[b].bounds, &primitive_info[i].bounds);
                    }

                    let mut cost = [0.0; N_BUCKETS - 1];

                    for i in 0..(N_BUCKETS - 1) {
                        let mut b0 = Bounds3::empty();
                        let mut b1 = Bounds3::empty();
                        let mut count0 = 0;
                        let mut count1 = 0;

                        for j in 0..i {
                            b0 = Bounds3::union(&b0, &buckets[j].bounds);
                            count0 += buckets[j].count;
                        }
                        for j in (i + 1)..N_BUCKETS {
                            b1 = Bounds3::union(&b1, &buckets[j].bounds);
                            count1 += buckets[j].count;
                        }
                        cost[i] = 1.0
                            + (count0 as f32 * b0.surface_area()
                                + count1 as f32 * b1.surface_area())
                                / bounds.surface_area();
                    }

                    let mut min_cost = cost[0];
                    let mut min_cost_split_bucket = 0usize;
                    for i in 1..(N_BUCKETS - 1) {
                        if cost[i] < min_cost {
                            min_cost = cost[i];
                            min_cost_split_bucket = i;
                        }
                    }

                    let leaf_cost = num_prims as f32;
                    if num_prims > *max_prims_in_node || min_cost < leaf_cost {
                        let p_mid =
                            primitive_info[start..end]
                                .iter_mut()
                                .partition_in_place(|pi| {
                                    let mut b = (N_BUCKETS as f32
                                        * centroid_bounds.offset(&pi.centroid)[dim])
                                        as usize;
                                    if b == N_BUCKETS {
                                        b = N_BUCKETS - 1;
                                    }

                                    return b <= min_cost_split_bucket;
                                });
                        mid = start + p_mid;
                    } else {
                        let first_prim_offset = ordered_prims.len();
                        for i in start..end {
                            let prim_num = primitive_info[i].prim_num;
                            ordered_prims.push(Arc::clone(&primitives[prim_num]));
                        }

                        return Box::new(BVHBuildNode::new_leaf(
                            first_prim_offset,
                            num_prims,
                            bounds,
                        ));
                    }
                }

                return Box::new(BVHBuildNode::new_interior(
                    dim,
                    BVH::recursive_build(
                        primitive_info,
                        max_prims_in_node,
                        start,
                        mid,
                        total_size,
                        ordered_prims,
                        primitives,
                    ),
                    BVH::recursive_build(
                        primitive_info,
                        max_prims_in_node,
                        mid,
                        end,
                        total_size,
                        ordered_prims,
                        primitives,
                    ),
                ));
            }
        }
    }

    fn flatten_bvh_tree(
        node: &Box<BVHBuildNode>,
        mut linear_nodes: &mut Box<[std::mem::MaybeUninit<LinearBVHNode>]>,
        mut offset: &mut usize,
    ) -> usize {
        let my_offset = *offset;
        *offset += 1;

        if node.num_prims > 0 {
            unsafe {
                linear_nodes[my_offset].as_mut_ptr().write(LinearBVHNode {
                    bounds: node.bounds,
                    offset: LinearBVHOffset {
                        primitives_offset: node.first_prim_offset as u32,
                    },
                    num_prims: node.num_prims as u16,
                    axis: 0,
                });
            }
        } else {
            BVH::flatten_bvh_tree(
                &node.children[0].as_ref().unwrap(),
                &mut linear_nodes,
                &mut offset,
            );
            let second_offset = BVH::flatten_bvh_tree(
                &node.children[1].as_ref().unwrap(),
                &mut linear_nodes,
                &mut offset,
            );

            unsafe {
                linear_nodes[my_offset].as_mut_ptr().write(LinearBVHNode {
                    bounds: node.bounds,
                    offset: LinearBVHOffset {
                        second_child_offset: second_offset as u32,
                    },
                    num_prims: 0,
                    axis: node.split_axis as u8,
                });
            }
        }

        my_offset
    }
}

impl Primitive for BVH {
    fn intersect<'a>(&'a self, r: &Ray, mut isect: &mut SurfaceInteraction<'a>) -> bool {
        if self.nodes.is_empty() {
            return false;
        }

        let mut hit = false;
        let inv_dir = na::Vector3::new(1.0f32 / r.d.x, 1.0f32 / r.d.y, 1.0f32 / r.d.z);
        let dir_is_neg = [inv_dir.x < 0.0, inv_dir.y < 0.0, inv_dir.z < 0.0];

        let mut to_visit_offset = 0;
        let mut curr_node_idx = 0;
        let mut nodes_to_visit = [0; 128];
        loop {
            let node = &self.nodes[curr_node_idx];

            if node.bounds.intersect_p_precomp(r, &inv_dir, &dir_is_neg) {
                if node.num_prims > 0 {
                    for i in 0..node.num_prims {
                        unsafe {
                            if self.primitives[node.offset.primitives_offset as usize + i as usize]
                                .intersect(r, &mut isect)
                            {
                                hit = true;
                            }
                        }
                    }

                    if to_visit_offset == 0 {
                        break;
                    }
                    to_visit_offset -= 1;
                    curr_node_idx = nodes_to_visit[to_visit_offset];
                } else {
                    if dir_is_neg[node.axis as usize] {
                        nodes_to_visit[to_visit_offset] = curr_node_idx + 1;
                        unsafe {
                            curr_node_idx = node.offset.second_child_offset as usize;
                        }
                    } else {
                        unsafe {
                            nodes_to_visit[to_visit_offset] =
                                node.offset.second_child_offset as usize;
                        }
                        curr_node_idx += 1;
                    }
                    to_visit_offset += 1;
                }
            } else {
                if to_visit_offset == 0 {
                    break;
                }
                to_visit_offset -= 1;
                curr_node_idx = nodes_to_visit[to_visit_offset];
            }
        }

        hit
    }

    fn intersect_p(&self, r: &Ray) -> bool {
        if self.nodes.is_empty() {
            return false;
        }

        let inv_dir = na::Vector3::new(1.0f32 / r.d.x, 1.0f32 / r.d.y, 1.0f32 / r.d.z);
        let dir_is_neg = [inv_dir.x < 0.0, inv_dir.y < 0.0, inv_dir.z < 0.0];

        let mut to_visit_offset = 0;
        let mut curr_node_idx = 0;
        let mut nodes_to_visit = [0; 128];
        loop {
            let node = &self.nodes[curr_node_idx];

            if node.bounds.intersect_p_precomp(r, &inv_dir, &dir_is_neg) {
                if node.num_prims > 0 {
                    for i in 0..node.num_prims {
                        unsafe {
                            if self.primitives[node.offset.primitives_offset as usize + i as usize]
                                .intersect_p(r)
                            {
                                return true;
                            }
                        }
                    }

                    if to_visit_offset == 0 {
                        break;
                    }
                    to_visit_offset -= 1;
                    curr_node_idx = nodes_to_visit[to_visit_offset];
                } else {
                    if dir_is_neg[node.axis as usize] {
                        nodes_to_visit[to_visit_offset] = curr_node_idx + 1;
                        unsafe {
                            curr_node_idx = node.offset.second_child_offset as usize;
                        }
                    } else {
                        unsafe {
                            nodes_to_visit[to_visit_offset] =
                                node.offset.second_child_offset as usize;
                        }
                        curr_node_idx += 1;
                    }
                    to_visit_offset += 1;
                }
            } else {
                if to_visit_offset == 0 {
                    break;
                }
                to_visit_offset -= 1;
                curr_node_idx = nodes_to_visit[to_visit_offset];
            }
        }

        false
    }

    fn world_bound(&self) -> Bounds3 {
        if self.nodes.is_empty() {
            Bounds3::empty()
        } else {
            self.nodes[0].bounds
        }
    }

    fn get_material(&self) -> &dyn super::material::SyncMaterial {
        unimplemented!()
    }

    fn compute_scattering_functions(
        &self,
        si: &mut SurfaceInteraction,
        mode: super::TransportMode,
    ) {
        unimplemented!()
    }
}
