#![feature(new_uninit)]
#![feature(iter_partition_in_place)]
#![feature(trait_alias)]

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

extern crate simba;

pub mod common;
pub mod headless;
pub mod pathtracer;
pub mod viewer;
