use crate::common::importer::mitsuba;
use crate::viewer::{Mesh, ViewerScene};

impl ViewerScene {
    pub fn from_mitsuba(log: &slog::Logger, scene: &mitsuba::Scene) -> Self {
        let mut meshes = vec![];

        Self { meshes }
    }
}
