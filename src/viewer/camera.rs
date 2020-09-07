use crate::common::Camera;
use winit::{dpi::LogicalPosition, event::*};

pub struct OrbitalCameraController {
    pivot: glm::Vec3,
    orbit_speed: f32,
    zoom_speed: f32,
    rotate_horizontal: f32,
    rotate_vertical: f32,
    scroll: f32,
    log: slog::Logger,
}

impl OrbitalCameraController {
    pub fn new(log: &slog::Logger, pivot: glm::Vec3, orbit_speed: f32, zoom_speed: f32) -> Self {
        let log = log.new(o!("camera controller" => "orbital"));
        Self {
            pivot,
            orbit_speed,
            zoom_speed,
            rotate_horizontal: 0.0,
            rotate_vertical: 0.0,
            scroll: 0.0,
            log,
        }
    }

    pub fn process_mouse(&mut self, mouse_dx: f64, mouse_dy: f64) {
        self.rotate_horizontal = mouse_dx.to_radians() as f32;
        self.rotate_vertical = mouse_dy.to_radians() as f32;
    }

    pub fn process_scroll(&mut self, delta: &MouseScrollDelta) {
        self.scroll = match delta {
            // I'm assuming a line is about 100 pixels
            MouseScrollDelta::LineDelta(_, scroll) => scroll * 100.0,
            MouseScrollDelta::PixelDelta(LogicalPosition { y: scroll, .. }) => *scroll as f32,
        };
    }

    pub fn update_camera(&mut self, camera: &mut Camera, dt: std::time::Duration) {
        let dt = dt.as_secs_f32();

        let mut cam_pos = camera.cam_to_world.translation.vector - self.pivot; // relative to pivot

        let vert_axis = glm::cross(&cam_pos, &glm::vec3(0.0f32, 1.0f32, 0.0f32));
        let horz_axis = glm::cross(&cam_pos, &vert_axis);
        cam_pos = glm::rotate_vec3(
            &cam_pos,
            self.rotate_horizontal * self.orbit_speed * dt,
            &horz_axis,
        );
        cam_pos = glm::rotate_vec3(
            &cam_pos,
            self.rotate_vertical * self.orbit_speed * dt,
            &vert_axis,
        );
        cam_pos = glm::normalize(&cam_pos)
            * 0.01_f32.max(glm::length(&cam_pos) * (1.0 + self.scroll * self.zoom_speed * dt));

        cam_pos += &self.pivot; // retransform back to global frame
        camera.cam_to_world = na::Isometry3::look_at_rh(
            &na::Point3::from(cam_pos),
            &na::Point3::from(self.pivot),
            &glm::vec3(0.0, 1.0, 0.0),
        )
        .inverse();

        trace!(self.log, "camera is now at: {:?}", camera.cam_to_world);

        self.rotate_horizontal = 0.0;
        self.rotate_vertical = 0.0;
        self.scroll = 0.0;
    }
}
