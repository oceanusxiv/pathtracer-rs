pub struct TrowbridgeReitzDistribution {}

impl TrowbridgeReitzDistribution {
    pub fn d(&self, wh: &na::Vector3<f32>) -> f32 {
        0.0
    }

    pub fn lambda(&self, w: &na::Vector3<f32>) -> f32 {
        0.0
    }

    pub fn g1(&self, w: &na::Vector3<f32>) -> f32 {
        1.0 / (1.0 + self.lambda(&w))
    }

    pub fn g(&self, wo: &na::Vector3<f32>, wi: &na::Vector3<f32>) -> f32 {
        1.0 / (1.0 + self.lambda(&wo) + self.lambda(&wi))
    }

    pub fn sample_wh(&self, wo: &na::Vector3<f32>, u: &na::Point2<f32>) -> na::Vector3<f32> {
        na::Vector3::zeros()
    }

    pub fn pdf(&self, wo: &na::Vector3<f32>, wh: &na::Vector3<f32>) -> f32 {
        0.0
    }

    pub fn roughness_to_alpha(roughness: f32) -> f32 {
        let roughness = roughness.max(1e-3);
        let x = roughness.ln();
        1.62142
            + 0.819955 * x
            + 0.1734 * x * x
            + 0.0171201 * x * x * x
            + 0.000640711 * x * x * x * x
    }
}

pub type MicrofacetDistribution = TrowbridgeReitzDistribution;
