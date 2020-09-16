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
}

// TODO: implement this
pub struct LanczosSincFilter {
    radius: na::Vector2<f32>,
    inv_radius: na::Vector2<f32>,
}

impl LanczosSincFilter {
    pub fn new(radius: &na::Vector2<f32>) -> Self {
        Self {
            radius: *radius,
            inv_radius: na::Vector2::new(1. / radius.x, 1. / radius.y),
        }
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
    inv_radius: na::Vector2<f32>,
}

impl TriangleFilter {
    pub fn new(radius: &na::Vector2<f32>) -> Self {
        Self {
            radius: *radius,
            inv_radius: na::Vector2::new(1. / radius.x, 1. / radius.y),
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
