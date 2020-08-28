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

pub type Spectrum = RGBSpectrum;
