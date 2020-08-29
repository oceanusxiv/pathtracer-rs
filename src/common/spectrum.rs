use std::ops::{Add, AddAssign, Div, Mul};
use super::math::gamma_correct;

#[derive(Clone, Debug, Copy)]
pub struct RGBSpectrum {
    pub r: f32,
    pub g: f32,
    pub b: f32,
}

impl RGBSpectrum {
    pub fn new(c: f32) -> Self {
        RGBSpectrum { r: c, g: c, b: c }
    }

    pub fn to_image_rgb(&self) -> image::Rgb<u8> {
        image::Rgb([
            (gamma_correct(self.r) * 255.0 + 0.5).clamp(0.0, 255.0) as u8,
            (gamma_correct(self.g) * 255.0 + 0.5).clamp(0.0, 255.0) as u8,
            (gamma_correct(self.b) * 255.0 + 0.5).clamp(0.0, 255.0) as u8,
        ])
    }

    pub fn is_black(&self) -> bool {
        self.r == 0.0 && self.g == 0.0 && self.b == 0.0
    }
}

impl AddAssign for RGBSpectrum {
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

impl Add for RGBSpectrum {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            r: self.r + rhs.r,
            g: self.g + rhs.g,
            b: self.b + rhs.b,
        }
    }
}

impl Mul for RGBSpectrum {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            r: self.r * rhs.r,
            g: self.g * rhs.g,
            b: self.b * rhs.b,
        }
    }
}

impl Mul<Spectrum> for f32 {
    type Output = Spectrum;

    fn mul(self, rhs: Spectrum) -> Self::Output {
        rhs * self
    }
}

impl Mul<f32> for RGBSpectrum {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        Self {
            r: self.r * rhs,
            g: self.g * rhs,
            b: self.b * rhs,
        }
    }
}

impl Div<f32> for RGBSpectrum {
    type Output = Self;

    fn div(self, rhs: f32) -> Self::Output {
        Self {
            r: self.r / rhs,
            g: self.g / rhs,
            b: self.b / rhs,
        }
    }
}

pub type Spectrum = RGBSpectrum;
