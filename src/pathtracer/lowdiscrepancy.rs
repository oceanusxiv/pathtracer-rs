use super::{
    sampling::{shuffle, Random},
    sobolmatrices::{
        NUM_SOBOL_DIMENSIONS, SOBOL_MATRICES_32, SOBOL_MATRIX_SIZE, VD_C_SOBOL_MATRICES,
        VD_C_SOBOL_MATRICES_INV,
    },
};
use crate::common::math::ONE_MINUS_EPSILON;
use rand::Rng;

const C_SOBOL: [[u32; 32]; 2] = [
    [
        0x8000_0000_u32,
        0x4000_0000,
        0x2000_0000,
        0x1000_0000,
        0x0800_0000,
        0x0400_0000,
        0x0200_0000,
        0x0100_0000,
        0x0080_0000,
        0x0040_0000,
        0x0020_0000,
        0x0010_0000,
        0x80000,
        0x40000,
        0x20000,
        0x10000,
        0x8000,
        0x4000,
        0x2000,
        0x1000,
        0x800,
        0x400,
        0x200,
        0x100,
        0x80,
        0x40,
        0x20,
        0x10,
        0x8,
        0x4,
        0x2,
        0x1,
    ],
    [
        0x8000_0000_u32,
        0xc000_0000,
        0xa000_0000,
        0xf000_0000,
        0x8800_0000,
        0xcc00_0000,
        0xaa00_0000,
        0xff00_0000,
        0x8080_0000,
        0xc0c0_0000,
        0xa0a0_0000,
        0xf0f0_0000,
        0x8888_0000,
        0xcccc_0000,
        0xaaaa_0000,
        0xffff_0000,
        0x8000_8000,
        0xc000_c000,
        0xa000_a000,
        0xf000_f000,
        0x8800_8800,
        0xcc00_cc00,
        0xaa00_aa00,
        0xff00_ff00,
        0x8080_8080,
        0xc0c0_c0c0,
        0xa0a0_a0a0,
        0xf0f0_f0f0,
        0x8888_8888,
        0xcccc_cccc,
        0xaaaa_aaaa,
        0xffff_ffff,
    ],
];

const INV_1_2_32: f32 = hexf32!("0x1.p-32");

fn gray_code_sample_2d(
    c0: &[u32],
    c1: &[u32],
    n: u32,
    scramble: &na::Point2<i32>,
    p: &mut [na::Point2<f32>],
) {
    let mut v = [scramble.x as u32, scramble.y as u32];

    // for i in 0..n {
    //     p[i].x =
    // }
}

pub fn sobol_2d(
    n_samples_per_pixel_sample: usize,
    n_pixel_samples: usize,
    samples: &mut [na::Point2<f32>],
    rng: &mut Random,
) {
    let scrample = na::Point2::new(rng.gen::<i32>(), rng.gen::<i32>());
}

pub fn sobol_interval_to_index(m: u32, mut frame: u64, p: na::Point2<i32>) -> u64 {
    if m == 0 {
        return 0;
    }

    let m2 = m << 1;
    let mut index = frame << m2;

    let mut delta = 0;
    let mut c = 0;
    while frame != 0 {
        if frame & 1 != 0 {
            delta ^= VD_C_SOBOL_MATRICES[(m - 1) as usize][c];
        }
        frame >>= 1;
        c += 1;
    }

    let mut b = (((p.x as u32) << m) as u64 | (p.y as u64)) ^ delta;

    let mut c = 0;
    while b != 0 {
        if b & 1 != 0 {
            index ^= VD_C_SOBOL_MATRICES_INV[(m - 1) as usize][c];
        }
        b >>= 1;
        c += 1;
    }

    return index;
}

// for f32 only!
pub fn sobol_sample(mut a: i64, dimenson: usize, scramble: u64) -> f32 {
    let mut v = scramble as u32;
    debug_assert!(dimenson < NUM_SOBOL_DIMENSIONS as usize);

    let mut i = dimenson * SOBOL_MATRIX_SIZE as usize;
    while a != 0 {
        if a & 1 != 0 {
            v ^= SOBOL_MATRICES_32[i];
        }

        a >>= 1;
        i += 1;
    }

    ONE_MINUS_EPSILON.min(v as f32 * INV_1_2_32)
}
