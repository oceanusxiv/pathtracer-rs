#[macro_use]
extern crate slog;

use pathtracer_rs::*;
use slog::Drain;

fn main() {
    let decorator = slog_term::TermDecorator::new().build();
    let drain = slog_term::FullFormat::new(decorator).build().fuse();
    let drain = slog_async::Async::new(drain).build().fuse();
    let log = slog::Logger::root(drain.fuse(), o!());
    let scene_path;
    if cfg!(target_os = "windows") {
        scene_path = "C://Users/eric1/Downloads/cornell-box/scene.xml";
    } else if cfg!(target_os = "macos") {
        scene_path = "/Users/eric/Downloads/cornell-box/scene.xml";
    } else {
        scene_path = "/home/eric/Downloads/cornell-box/scene.xml";
    }
    info!(log, "openning scene: {:?}", scene_path);
    let pixel_samples_sqrt = 2;
    common::importer::mitsuba::from_mitsuba(&log, &scene_path, &common::DEFAULT_RESOLUTION);
}
