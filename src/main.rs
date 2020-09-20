#[macro_use]
extern crate slog;

#[macro_use]
extern crate anyhow;

#[macro_use]
extern crate maplit;

extern crate nalgebra as na;

use anyhow::Result;
use clap::clap_app;
use pathtracer_rs::*;
use slog::Drain;
use std::collections::HashSet;
use std::iter::FromIterator;
use std::path::Path;
use std::str::FromStr;

const MAX_DEPTH: i32 = 20;

fn parse_resolution(res_str: &str) -> Result<na::Vector2<f32>> {
    let xy = res_str.split("x").collect::<Vec<_>>();
    if xy.len() != 2 {
        Err(anyhow!("invalid resolution string"))
    } else {
        let x = xy[0].parse::<f32>()?;
        let y = xy[1].parse::<f32>()?;

        Ok(na::Vector2::new(x, y))
    }
}

fn main() {
    let matches = clap_app!(pathtracer_rs =>
        (version: "1.0")
        (author: "Eric F. <eric1221bday@gmail.com>")
        (about: "Rust path tracer")
        (@arg SCENE: +required "Sets the input scene to use")
        (@arg output: -o --output +takes_value +required "Sets the output directory to save renders at")
        (@arg samples: -s --samples default_value("1") "Number of samples path tracer to take per pixel (sampler dependent)")
        (@arg resolution: -r --resolution +takes_value "Resolution of the window")
        (@arg camera_controller: -c --camera default_value("orbit") "Camera movement type")
        (@arg max_depth: -d --max_depth default_value("15") "Maximum ray tracing depth")
        (@arg log_level: -l --log_level default_value("INFO") "Application wide log level")
        (@arg module_log: -m --module_log default_value("all") "Module names to log, (all for every module)")
        (@arg default_lights: --default_lights "Add default lights into the scene")
        (@arg verbose: -v --verbose "Print test information verbosely")
    )
    .get_matches();

    let init_log_level = slog::Level::from_str(matches.value_of("log_level").unwrap())
        .unwrap_or_else(|()| slog::Level::Info);
    let allowed_modules_str = matches.value_of("module_log").unwrap();
    let allowed_modules = if allowed_modules_str == "all" {
        None
    } else {
        Some(
            hashmap! {String::from("module") => HashSet::<String>::from_iter(allowed_modules_str.split(",").map(|s| String::from(s)).into_iter())},
        )
    };
    let drain = common::new_drain(init_log_level, &allowed_modules);
    let drain = slog_atomic::AtomicSwitch::new(drain);
    let ctrl = drain.ctrl();
    let log = slog::Logger::root(drain.fuse(), o!());

    let scene_path = matches.value_of("SCENE").unwrap();
    let output_path = Path::new(matches.value_of("output").unwrap()).join("render.png");
    let pixel_samples = matches
        .value_of("samples")
        .unwrap()
        .parse::<usize>()
        .unwrap();
    let resolution = if let Some(res_str) = matches.value_of("resolution") {
        parse_resolution(&res_str).unwrap_or_else(|_| {
            warn!(
                log,
                "failed parsing resolution string, falling back to default resolution"
            );
            *common::DEFAULT_RESOLUTION
        })
    } else {
        *common::DEFAULT_RESOLUTION
    };
    let max_depth = matches
        .value_of("max_depth")
        .unwrap()
        .parse::<i32>()
        .unwrap_or_else(|_| {
            warn!(
                log,
                "failed parsing max depth, falling back to default max depth"
            );
            MAX_DEPTH
        });

    let camera_controller_type = matches.value_of("camera_controller").unwrap();

    let default_lights = matches.is_present("default_lights");

    let (camera, render_scene, viewer_scene) =
        common::importer::import(&log, &scene_path, &resolution, default_lights);
    let sampler = pathtracer::sampler::SamplerBuilder::new(
        &log,
        pixel_samples,
        &camera.film.get_sample_bounds(),
    );
    let mut integrator = pathtracer::integrator::PathIntegrator::new(&log, sampler, max_depth);
    integrator.preprocess(&render_scene);

    debug!(log, "camera starting at: {:?}", camera.cam_to_world);

    viewer::run(
        log,
        &resolution,
        &viewer_scene,
        render_scene,
        camera,
        camera_controller_type,
        integrator,
        output_path,
        ctrl,
        pixel_samples,
        max_depth,
        init_log_level,
        allowed_modules,
    );
}
