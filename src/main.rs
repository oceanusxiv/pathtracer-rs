extern crate nalgebra_glm as glm;

mod common;
//mod gui;
mod pathtracer;

fn main() -> Result<(), failure::Error> {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Warn)
        .filter_module("pathtracer_rs", log::LevelFilter::Info)
        .init();

    let world = common::World::from_gltf("C:/Users/eric1/Downloads/Buggy.glb");
    //    gui::run_gui();
    Ok(())
}
