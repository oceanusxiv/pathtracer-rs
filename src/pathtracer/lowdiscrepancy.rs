use super::{
    sampling::{shuffle, Random},
    sobolmatrices::{
        NUM_SOBOL_DIMENSIONS, SOBOL_MATRICES_32, SOBOL_MATRIX_SIZE, VD_C_SOBOL_MATRICES,
        VD_C_SOBOL_MATRICES_INV,
    },
};
use crate::common::math::ONE_MINUS_EPSILON;
use rand::Rng;

const INV_1_2_32: f32 = hexf32!("0x1.p-32");

pub fn sobol_interval_to_index(m: u32, mut frame: u64, p: &na::Point2<i32>) -> u64 {
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
pub fn sobol_sample(mut a: i64, dimension: usize, scramble: u64) -> f32 {
    let mut v = scramble as u32;
    debug_assert!(dimension < NUM_SOBOL_DIMENSIONS as usize);

    let mut i = dimension * SOBOL_MATRIX_SIZE as usize;
    while a != 0 {
        if a & 1 != 0 {
            v ^= SOBOL_MATRICES_32[i];
        }

        a >>= 1;
        i += 1;
    }

    ONE_MINUS_EPSILON.min(v as f32 * INV_1_2_32)
}
