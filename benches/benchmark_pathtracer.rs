#[macro_use]
extern crate slog;

extern crate nalgebra as na;

use criterion::*;
use itertools::izip;
use pathtracer_rs::pathtracer::sampling::cosine_sample_hemisphere;
use pathtracer_rs::*;
use rand::prelude::*;
use rand::rngs::SmallRng;

fn bench_render(c: &mut Criterion) {
    // Starting the Tracy client is necessary before any invoking any of its APIs
    #[cfg(feature = "enable_profiling")]
    profiling::tracy_client::Client::start();

    // Good to call this on any threads that are created to get clearer profiling results
    #[cfg(feature = "enable_profiling")]
    profiling::register_thread!("Main Thread");

    let mut group = c.benchmark_group("benchmark-render");

    let drain = slog::Discard;
    let log = slog::Logger::root(drain, o!());
    let scene_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("data/cornell-box.xml");
    let scene_path = scene_path.to_str().unwrap();

    let pixel_samples = 16;
    let (mut camera, render_scene, _) =
        common::importer::import(&log, &scene_path, &common::DEFAULT_RESOLUTION, false);
    let sampler = pathtracer::sampler::SamplerBuilder::new(
        &log,
        pixel_samples,
        &camera.film.get_sample_bounds(),
    );
    let mut integrator = pathtracer::integrator::PathIntegrator::new(&log, sampler, 5, false);
    integrator.preprocess(&render_scene);

    group.sampling_mode(SamplingMode::Flat).sample_size(10);
    group.bench_function("bench_render", |b| {
        b.iter(|| integrator.render(&mut camera, &render_scene))
    });
    group.finish();
}

fn bench_bounds(c: &mut Criterion) {
    let mut group = c.benchmark_group("benchmark-bounds");

    let between = rand::distributions::Uniform::from(0.0..1.0);
    let mut rng = SmallRng::seed_from_u64(1);
    let mut rs = vec![];
    let mut inv_dirs = vec![];
    let mut dir_is_negs = vec![];
    for _ in 0..1000 {
        let r = common::ray::Ray {
            o: nalgebra::Point3::new(-1.0, -1.0, -1.0),
            d: cosine_sample_hemisphere(&nalgebra::Point2::new(
                between.sample(&mut rng),
                between.sample(&mut rng),
            )),
            t_max: f32::INFINITY,
        };
        let inv_dir = nalgebra::Vector3::new(1.0f32 / r.d.x, 1.0f32 / r.d.y, 1.0f32 / r.d.z);
        let dir_is_neg = [inv_dir.x < 0.0, inv_dir.y < 0.0, inv_dir.z < 0.0];

        rs.push(r);
        inv_dirs.push(inv_dir);
        dir_is_negs.push(dir_is_neg);
    }

    let bounds = common::bounds::Bounds3 {
        p_min: nalgebra::Point3::origin(),
        p_max: nalgebra::Point3::new(1.0, 1.0, 1.0),
    };

    group.bench_function("bench_intersect_p", |b| {
        b.iter(|| {
            for r in &rs {
                bounds.intersect_p(&r);
            }
        })
    });

    group.bench_function("bench_intersect_p_precomp", |b| {
        b.iter(|| {
            for (r, inv_dir, dir_is_neg) in izip!(&rs, &inv_dirs, &dir_is_negs) {
                bounds.intersect_p_precomp(&r, &inv_dir, &dir_is_neg);
            }
        })
    });
    group.finish();
}

fn bench_samplers(c: &mut Criterion) {
    let mut group = c.benchmark_group("benchmark-samplers");

    let drain = slog::Discard;
    let log = slog::Logger::root(drain, o!());
    let pixel_samples_sqrt = 32;
    let mut sampler =
        pathtracer::sampler::stratified::StratifiedSamplerBuilder::new(&log, pixel_samples_sqrt, 8)
            .with_seed(10)
            .build();

    group.bench_function("bench_stratified_sampler_start_pixel", |b| {
        b.iter(|| sampler.start_pixel(&na::Point2::new(0, 0)))
    });
    group.finish();
}

criterion_group!(benches, bench_bounds, bench_render, bench_samplers);
criterion_main!(benches);
