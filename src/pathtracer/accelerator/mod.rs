use super::primitive::Primitive;
use super::Bounds3;
use std::rc::Rc;

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
    pub fn init_leaf(first: usize, n: usize, b: Bounds3) -> Self {
        BVHBuildNode {
            bounds: b,
            children: [None, None],
            split_axis: 0,
            first_prim_offset: first,
            num_prims: n,
        }
    }

    pub fn init_interior() {}
}

struct LinearBVHNode {}

pub struct BVH {
    primitives: Vec<Rc<dyn Primitive>>,
    nodes: Box<[LinearBVHNode]>,
}

impl BVH {
    pub fn new(primitives: Vec<Rc<dyn Primitive>>) -> Self {
        let mut primitive_info = Vec::<BVHPrimitiveInfo>::with_capacity(primitives.len());

        for i in 0..primitives.len() {
            primitive_info.push(BVHPrimitiveInfo::new(i, primitives[i].world_bound()))
        }

        let mut total_nodes = 0usize;
        let mut ordered_prims = Vec::<Rc<dyn Primitive>>::with_capacity(primitives.len());

        let root = BVH::recursive_build(
            primitive_info,
            0,
            primitives.len(),
            &mut total_nodes,
            &mut ordered_prims,
            &primitives,
        );

        let mut nodes = std::iter::repeat_with(|| std::mem::MaybeUninit::<LinearBVHNode>::uninit()) // you can use `repeat` in case `T: Copy`
            .take(primitives.len())
            .collect::<Box<[_]>>();
        let mut offset = 0usize;
        BVH::flatten_bvh_tree(root, &mut nodes, &mut offset);

        let nodes = unsafe { Box::from_raw(Box::into_raw(nodes) as *mut [LinearBVHNode]) };

        BVH {
            primitives: ordered_prims,
            nodes,
        }
    }

    fn recursive_build(
        primitive_info: Vec<BVHPrimitiveInfo>,
        start: usize,
        end: usize,
        total_size: &mut usize,
        ordered_prims: &mut Vec<Rc<dyn Primitive>>,
        primitives: &Vec<Rc<dyn Primitive>>,
    ) -> Box<BVHBuildNode> {

        Box::new(BVHBuildNode::init_leaf(0, 0, Bounds3::empty()))
    }

    fn flatten_bvh_tree(
        node: Box<BVHBuildNode>,
        linear_nodes: &mut Box<[std::mem::MaybeUninit<LinearBVHNode>]>,
        offset: &mut usize,
    ) -> usize {
        1usize
    }
}

impl Primitive for BVH {
    fn intersect(&self, r: &super::Ray) -> Option<super::SurfaceInteraction> {
        todo!()
    }

    fn world_bound(&self) -> Bounds3 {
        todo!()
    }
}
