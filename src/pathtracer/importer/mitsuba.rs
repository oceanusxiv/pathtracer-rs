use crate::pathtracer::{accelerator, light::SyncLight, primitive::SyncPrimitive, RenderScene};
use std::sync::Arc;

impl RenderScene {
    pub fn from_mitsuba(log: &slog::Logger) -> Self {
        let mut primitives: Vec<Arc<dyn SyncPrimitive>> = Vec::new();
        let mut lights: Vec<Arc<dyn SyncLight>> = Vec::new();

        let bvh = Box::new(accelerator::BVH::new(log, primitives, &4)) as Box<dyn SyncPrimitive>;

        Self { scene: bvh, lights }
    }
}
