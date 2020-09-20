#![feature(new_uninit)]
#![feature(slice_partition_at_index)]
#![feature(iter_partition_in_place)]
#![feature(clamp)]

#[macro_use]
extern crate bitflags;

#[macro_use]
extern crate hexf;

#[macro_use]
extern crate slog;

#[macro_use]
extern crate serde_derive;

extern crate nalgebra as na;
extern crate nalgebra_glm as glm;

pub mod common;
pub mod headless;
pub mod pathtracer;
pub mod viewer;
