use crate::{
    viewer::{Mesh, ViewerScene},
};
use std::collections::HashMap;

impl ViewerScene {
    pub fn from_mitsuba(
        log: &slog::Logger,
        path: &str
    ) -> Self {
        let mut meshes = vec![];

        Self { meshes }
    }
}