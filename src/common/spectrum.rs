use super::math::{gamma_correct, inverse_gamma_correct};
use num::Zero;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub};

// TODO: think about making this use vectors inside
#[derive(Clone, Debug, Copy)]
pub struct RGBSpectrum(na::Vector3<f32>);

impl RGBSpectrum {
    pub fn new(c: f32) -> Self {
        Self(na::Vector3::new(c, c, c))
    }

    pub fn from_floats(r: f32, g: f32, b: f32) -> Self {
        Self(na::Vector3::new(r, g, b))
    }

    pub fn r(&self) -> f32 {
        self.0[0]
    }
    pub fn g(&self) -> f32 {
        self.0[1]
    }
    pub fn b(&self) -> f32 {
        self.0[2]
    }

    pub fn from_image_rgb(color: &image::Rgb<u8>, gamma: bool) -> Self {
        if gamma {
            Self(na::Vector3::new(
                inverse_gamma_correct(color[0] as f32 / 255.0),
                inverse_gamma_correct(color[1] as f32 / 255.0),
                inverse_gamma_correct(color[2] as f32 / 255.0),
            ))
        } else {
            Self(na::Vector3::new(
                color[0] as f32 / 255.0,
                color[1] as f32 / 255.0,
                color[2] as f32 / 255.0,
            ))
        }
    }

    pub fn from_image_rgba(color: &image::Rgba<u8>, gamma: bool) -> Self {
        if gamma {
            Self(na::Vector3::new(
                inverse_gamma_correct(color[0] as f32 / 255.0),
                inverse_gamma_correct(color[1] as f32 / 255.0),
                inverse_gamma_correct(color[2] as f32 / 255.0),
            ))
        } else {
            Self(na::Vector3::new(
                color[0] as f32 / 255.0,
                color[1] as f32 / 255.0,
                color[2] as f32 / 255.0,
            ))
        }
    }

    pub fn from_image_rgb_f32(color: &image::Rgb<f32>) -> Self {
        Self(na::Vector3::new(color[0], color[1], color[2]))
    }

    pub fn from_slice_4(slice: &[f32; 4], gamma: bool) -> Self {
        if gamma {
            Self(na::Vector3::new(
                inverse_gamma_correct(slice[0]),
                inverse_gamma_correct(slice[1]),
                inverse_gamma_correct(slice[2]),
            ))
        } else {
            Self(na::Vector3::new(slice[0], slice[1], slice[2]))
        }
    }

    pub fn from_slice_3(slice: &[f32; 3], gamma: bool) -> Self {
        if gamma {
            Self(na::Vector3::new(
                inverse_gamma_correct(slice[0]),
                inverse_gamma_correct(slice[1]),
                inverse_gamma_correct(slice[2]),
            ))
        } else {
            Self(na::Vector3::new(slice[0], slice[1], slice[2]))
        }
    }

    pub fn to_image_rgb(&self) -> image::Rgb<u8> {
        image::Rgb([
            (gamma_correct(self.r()) * 255.0 + 0.5).clamp(0.0, 255.0) as u8,
            (gamma_correct(self.g()) * 255.0 + 0.5).clamp(0.0, 255.0) as u8,
            (gamma_correct(self.b()) * 255.0 + 0.5).clamp(0.0, 255.0) as u8,
        ])
    }

    pub fn to_image_rgba(&self) -> image::Rgba<u8> {
        image::Rgba([
            (gamma_correct(self.r()) * 255.0 + 0.5).clamp(0.0, 255.0) as u8,
            (gamma_correct(self.g()) * 255.0 + 0.5).clamp(0.0, 255.0) as u8,
            (gamma_correct(self.b()) * 255.0 + 0.5).clamp(0.0, 255.0) as u8,
            255,
        ])
    }

    pub fn is_black(&self) -> bool {
        self.is_zero()
    }

    pub fn has_nan(&self) -> bool {
        self.r().is_nan() || self.g().is_nan() || self.b().is_nan()
    }

    pub fn y(&self) -> f32 {
        const Y_WEIGHT: [f32; 3] = [0.212671, 0.715160, 0.072169];
        self.r() * Y_WEIGHT[0] + self.g() * Y_WEIGHT[1] + self.b() * Y_WEIGHT[2]
    }

    pub fn max_component_value(&self) -> f32 {
        self.r().max(self.g()).max(self.b())
    }

    pub fn sqrt(&self) -> Self {
        Self(na::Vector3::new(
            self.r().sqrt(),
            self.g().sqrt(),
            self.b().sqrt(),
        ))
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
        Self(self.0 + rhs.0)
    }
}

impl Add<f32> for RGBSpectrum {
    type Output = Self;

    fn add(self, rhs: f32) -> Self::Output {
        Self::from_floats(self.r() + rhs, self.g() + rhs, self.b() + rhs)
    }
}

impl Sub for RGBSpectrum {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl Sub<f32> for RGBSpectrum {
    type Output = Self;

    fn sub(self, rhs: f32) -> Self::Output {
        Self::from_floats(self.r() - rhs, self.g() - rhs, self.b() - rhs)
    }
}

impl Mul for RGBSpectrum {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0.component_mul(&rhs.0))
    }
}

impl MulAssign for RGBSpectrum {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl MulAssign<f32> for RGBSpectrum {
    fn mul_assign(&mut self, rhs: f32) {
        *self = *self * rhs
    }
}

impl Mul<RGBSpectrum> for f32 {
    type Output = Spectrum;

    fn mul(self, rhs: Spectrum) -> Self::Output {
        rhs * self
    }
}

impl Mul<f32> for RGBSpectrum {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        Self::from_floats(self.r() * rhs, self.g() * rhs, self.b() * rhs)
    }
}

impl Div for RGBSpectrum {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        Self(self.0.component_div(&rhs.0))
    }
}

impl Div<f32> for RGBSpectrum {
    type Output = Self;

    fn div(self, rhs: f32) -> Self::Output {
        Self::from_floats(self.r() / rhs, self.g() / rhs, self.b() / rhs)
    }
}

impl DivAssign<f32> for RGBSpectrum {
    fn div_assign(&mut self, rhs: f32) {
        *self = *self / rhs
    }
}

impl PartialEq for RGBSpectrum {
    fn eq(&self, other: &Self) -> bool {
        self == other
    }
}

impl num::Zero for RGBSpectrum {
    fn zero() -> Self {
        Self::new(0.0)
    }

    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

impl num::One for RGBSpectrum {
    fn one() -> Self {
        Self::new(1.0)
    }
}

pub type Spectrum = RGBSpectrum;
