#[macro_use]
extern crate slog;

use criterion::*;
use pathtracer_rs::*;

fn bench(c: &mut Criterion) {
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

    group.bench_function("intersect_p_precomp", |b| {
        b.iter(|| bounds.intersect_p_precomp(&r, &inv_dir, &dir_is_neg))
    });
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
