#[macro_use]
extern crate slog;

use criterion::*;
use pathtracer_rs::*;

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("benchmark-render");

    let drain = slog::Discard;
    let log = slog::Logger::root(drain, o!());
    let scene_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("data/cornell-box.xml");
    let scene_path = scene_path.to_str().unwrap();

    let pixel_samples_sqrt = 4;
    let (mut camera, render_scene, _) =
        common::importer::import(&log, &scene_path, &common::DEFAULT_RESOLUTION, false);
    let sampler = pathtracer::sampler::SamplerBuilder::new(
        &log,
        pixel_samples_sqrt,
        pixel_samples_sqrt,
        true,
        8,
    );
    let mut integrator = pathtracer::integrator::PathIntegrator::new(&log, sampler, 5);
    integrator.preprocess(&render_scene);

    group.sampling_mode(SamplingMode::Flat).sample_size(10);
    group.bench_function("render", |b| {
        b.iter(|| integrator.render(&mut camera, &render_scene))
    });
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
