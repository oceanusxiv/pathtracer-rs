use super::math::{gamma_correct, inverse_gamma_correct};
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Sub};

#[derive(Clone, Debug, Copy)]
pub struct RGBSpectrum {
    pub r: f32,
    pub g: f32,
    pub b: f32,
}

impl RGBSpectrum {
    pub fn new(c: f32) -> Self {
        Self { r: c, g: c, b: c }
    }

    pub fn from_image_rgb(color: &image::Rgb<u8>, gamma: bool) -> Self {
        if gamma {
            Self {
                r: inverse_gamma_correct(color[0] as f32 / 255.0),
                g: inverse_gamma_correct(color[1] as f32 / 255.0),
                b: inverse_gamma_correct(color[2] as f32 / 255.0),
            }
        } else {
            Self {
                r: color[0] as f32 / 255.0,
                g: color[1] as f32 / 255.0,
                b: color[2] as f32 / 255.0,
            }
        }
    }

    pub fn from_slice(slice: &[f32; 4], gamma: bool) -> Self {
        if gamma {
            Self {
                r: inverse_gamma_correct(slice[0]),
                g: inverse_gamma_correct(slice[1]),
                b: inverse_gamma_correct(slice[2]),
            }
        } else {
            Self {
                r: slice[0],
                g: slice[1],
                b: slice[2],
            }
        }
    }

    pub fn to_image_rgb(&self) -> image::Rgb<u8> {
        image::Rgb([
            (gamma_correct(self.r) * 255.0 + 0.5).clamp(0.0, 255.0) as u8,
            (gamma_correct(self.g) * 255.0 + 0.5).clamp(0.0, 255.0) as u8,
            (gamma_correct(self.b) * 255.0 + 0.5).clamp(0.0, 255.0) as u8,
        ])
    }

    pub fn to_image_rgba(&self) -> image::Rgba<u8> {
        image::Rgba([
            (gamma_correct(self.r) * 255.0 + 0.5).clamp(0.0, 255.0) as u8,
            (gamma_correct(self.g) * 255.0 + 0.5).clamp(0.0, 255.0) as u8,
            (gamma_correct(self.b) * 255.0 + 0.5).clamp(0.0, 255.0) as u8,
            255,
        ])
    }

    pub fn is_black(&self) -> bool {
        self.r == 0.0 && self.g == 0.0 && self.b == 0.0
    }

    pub fn has_nan(&self) -> bool {
        self.r.is_nan() || self.g.is_nan() || self.b.is_nan()
    }

    pub fn y(&self) -> f32 {
        const Y_WEIGHT: [f32; 3] = [0.212671, 0.715160, 0.072169];
        self.r * Y_WEIGHT[0] + self.g * Y_WEIGHT[1] + self.b * Y_WEIGHT[2]
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

impl Sub for RGBSpectrum {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            r: self.r - rhs.r,
            g: self.g - rhs.g,
            b: self.b - rhs.b,
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

impl MulAssign<f32> for Spectrum {
    fn mul_assign(&mut self, rhs: f32) {
        *self = *self * rhs
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

impl PartialEq for RGBSpectrum {
    fn eq(&self, other: &Self) -> bool {
        self.r == other.r && self.g == other.g && self.b == other.b
    }
}

impl num::Zero for RGBSpectrum {
    fn zero() -> Self {
        Self::new(0.0)
    }

    fn is_zero(&self) -> bool {
        self.r == 0.0 && self.g == 0.0 && self.b == 0.0
    }
}

pub type Spectrum = RGBSpectrum;
