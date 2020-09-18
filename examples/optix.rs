use pathtracer_rs::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    pathtracer::accelerator::optix::OptixAccelerator::new()?;

    Ok(())
}
