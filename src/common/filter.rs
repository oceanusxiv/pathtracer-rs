use ambassador::{delegatable_trait, Delegate};

#[delegatable_trait]
pub trait FilterInterface {
    fn evaluate(&self, p: &na::Point2<f32>) -> f32;
    fn radius(&self) -> &na::Vector2<f32>;
}

#[derive(Delegate)]
#[delegate(FilterInterface)]
pub enum Filter {
    Sinc(LanczosSincFilter),
    Triangle(TriangleFilter),
    Guassian(GuassianFilter),
}

// TODO: implement this
pub struct LanczosSincFilter {
    radius: na::Vector2<f32>,
}

impl LanczosSincFilter {
    pub fn new(radius: &na::Vector2<f32>) -> Self {
        Self { radius: *radius }
    }
}

impl FilterInterface for LanczosSincFilter {
    fn evaluate(&self, p: &na::Point2<f32>) -> f32 {
        todo!()
    }

    fn radius(&self) -> &na::Vector2<f32> {
        &self.radius
    }
}

pub struct TriangleFilter {
    radius: na::Vector2<f32>,
}

impl TriangleFilter {
    pub fn new() -> Self {
        const RADIUS: f32 = 1.0;
        Self {
            radius: na::Vector2::new(RADIUS, RADIUS),
        }
    }
}

impl FilterInterface for TriangleFilter {
    fn evaluate(&self, p: &na::Point2<f32>) -> f32 {
        0.0f32.max(self.radius.x - p.x.abs()) * 0.0f32.max(self.radius.y - p.y.abs())
    }

    fn radius(&self) -> &na::Vector2<f32> {
        &self.radius
    }
}

pub struct GuassianFilter {
    alpha: f32,
    exp: f32,
    radius: na::Vector2<f32>,
}

impl GuassianFilter {
    pub fn new(alpha: f32) -> Self {
        let radius = 2.0;
        Self {
            alpha,
            exp: (-alpha * radius * radius).exp(),
            radius: na::Vector2::new(radius, radius),
        }
    }

    fn guassian(&self, d: f32, expv: f32) -> f32 {
        0.0f32.max((-self.alpha * d * d).exp() - expv)
    }
}

impl FilterInterface for GuassianFilter {
    fn evaluate(&self, p: &na::Point2<f32>) -> f32 {
        self.guassian(p.x, self.exp) * self.guassian(p.y, self.exp)
    }

    fn radius(&self) -> &na::Vector2<f32> {
        &self.radius
    }
}
