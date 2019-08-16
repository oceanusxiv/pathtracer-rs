mod common;
mod gui;

fn main() -> Result<(), failure::Error> {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Warn)
        .filter_module("pathtracer_rs", log::LevelFilter::Info)
        .init();

    gui::run_gui();
    Ok(())
}
