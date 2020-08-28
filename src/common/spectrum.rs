use std::ops::{Add, AddAssign};

#[derive(Clone, Debug, Copy)]
pub struct RGBSpectrum {
    pub r: f32,
    pub g: f32,
    pub b: f32,
}

impl RGBSpectrum {
    pub fn to_image_rgb(&self) -> image::Rgb<u8> {
        image::Rgb([
            (self.r * 255.0) as u8,
            (self.g * 255.0) as u8,
            (self.b * 255.0) as u8,
        ])
    }
}

impl AddAssign for RGBSpectrum {
    fn add_assign(&mut self, other: Self) {
        *self = Self {
            r: self.r + other.r,
            g: self.g + other.g,
            b: self.b + other.b,
        };
    }
}

impl Add for RGBSpectrum {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            r: self.r + other.r,
            g: self.g + other.g,
            b: self.b + other.b,
        }
    }
}

pub type Spectrum = RGBSpectrum;
