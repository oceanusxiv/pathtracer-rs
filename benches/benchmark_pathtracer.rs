#[macro_use]
extern crate slog;

use criterion::*;
use pathtracer_rs::*;

fn bench_render(c: &mut Criterion) {
    let mut group = c.benchmark_group("benchmark-render");

    let drain = slog::Discard;
    let log = slog::Logger::root(drain, o!());
    let scene_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("data/cornell-box.xml");
    let scene_path = scene_path.to_str().unwrap();

    let pixel_samples_sqrt = 4;
    let (mut camera, render_scene, _) =
        common::importer::import(&log, &scene_path, &common::DEFAULT_RESOLUTION, false);
    let sampler = pathtracer::sampler::SamplerBuilder::new(&log, pixel_samples_sqrt, 8);
    let mut integrator = pathtracer::integrator::PathIntegrator::new(&log, sampler, 5);
    integrator.preprocess(&render_scene);

    group.sampling_mode(SamplingMode::Flat).sample_size(10);
    group.bench_function("bench_render", |b| {
        b.iter(|| integrator.render(&mut camera, &render_scene))
    });
    group.finish();
}

fn bench_bounds(c: &mut Criterion) {
    let mut group = c.benchmark_group("benchmark-bounds");

    let r = common::ray::Ray {
        o: nalgebra::Point3::origin(),
        d: nalgebra::Vector3::new(1.0, 1.0, 1.0),
        t_max: f32::INFINITY,
    };
    let inv_dir = nalgebra::Vector3::new(1.0f32 / r.d.x, 1.0f32 / r.d.y, 1.0f32 / r.d.z);
    let dir_is_neg = [inv_dir.x < 0.0, inv_dir.y < 0.0, inv_dir.z < 0.0];
    let bounds = common::bounds::Bounds3 {
        p_min: nalgebra::Point3::origin(),
        p_max: nalgebra::Point3::new(1.0, 1.0, 1.0),
    };

    group.bench_function("bench_bounds", |b| {
        b.iter(|| bounds.intersect_p_precomp(&r, &inv_dir, &dir_is_neg))
    });
    group.finish();
}

criterion_group!(benches, bench_bounds, bench_render);
criterion_main!(benches);
