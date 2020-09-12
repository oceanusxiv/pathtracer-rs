use crate::common::Camera;
use ambassador::{delegatable_trait, Delegate};
use winit::{dpi::LogicalPosition, event::*};

#[delegatable_trait]
pub trait CameraControllerInterface {
    fn process_mouse(&mut self, mouse_dx: f64, mouse_dy: f64);
    fn process_key(&mut self, _key: &VirtualKeyCode) -> bool {
        false
    }
    fn process_scroll(&mut self, _delta: &MouseScrollDelta) {}
    fn update_camera(&mut self, camera: &mut Camera, dt: std::time::Duration);
    fn require_mouse_press(&self) -> bool;
}

#[derive(Delegate)]
#[delegate(CameraControllerInterface)]
pub enum CameraController {
    Orbit(OrbitalCameraController),
    FirstPerson(FirstPersonCameraController),
}

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
}

impl CameraControllerInterface for OrbitalCameraController {
    fn process_mouse(&mut self, mouse_dx: f64, mouse_dy: f64) {
        self.rotate_horizontal = mouse_dx.to_radians() as f32;
        self.rotate_vertical = mouse_dy.to_radians() as f32;
    }

    fn process_scroll(&mut self, delta: &MouseScrollDelta) {
        self.scroll = match delta {
            // I'm assuming a line is about 100 pixels
            MouseScrollDelta::LineDelta(_, scroll) => scroll * 100.0,
            MouseScrollDelta::PixelDelta(LogicalPosition { y: scroll, .. }) => *scroll as f32,
        };
    }

    fn update_camera(&mut self, camera: &mut Camera, dt: std::time::Duration) {
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

    fn require_mouse_press(&self) -> bool {
        true
    }
}

pub struct FirstPersonCameraController {
    rotate_sensitivity: f32,
    move_sensitivity: f32,
    translation: na::Translation3<f32>,
    rotation: (f32, f32),
    spin: f32,
    log: slog::Logger,
}

impl FirstPersonCameraController {
    pub fn new(log: &slog::Logger, rotate_sensitivity: f32, move_sensitivity: f32) -> Self {
        let log = log.new(o!());
        Self {
            rotate_sensitivity,
            move_sensitivity,
            translation: na::Translation3::identity(),
            rotation: (0.0, 0.0),
            spin: 0.0,
            log,
        }
    }
}

impl CameraControllerInterface for FirstPersonCameraController {
    fn process_key(&mut self, key: &VirtualKeyCode) -> bool {
        match key {
            VirtualKeyCode::W => {
                self.translation.z = -self.move_sensitivity;
                true
            }
            VirtualKeyCode::A => {
                self.translation.x = -self.move_sensitivity;
                true
            }
            VirtualKeyCode::S => {
                self.translation.z = self.move_sensitivity;
                true
            }
            VirtualKeyCode::D => {
                self.translation.x = self.move_sensitivity;
                true
            }
            VirtualKeyCode::Z => {
                self.translation.y = self.move_sensitivity;
                true
            }
            VirtualKeyCode::X => {
                self.translation.y = -self.move_sensitivity;
                true
            }
            VirtualKeyCode::Q => {
                self.spin = self.move_sensitivity;
                true
            }
            VirtualKeyCode::E => {
                self.spin = -self.move_sensitivity;
                true
            }
            _ => false,
        }
    }

    fn process_mouse(&mut self, mouse_dx: f64, mouse_dy: f64) {
        self.rotation = (
            -mouse_dy.to_radians() as f32 * self.rotate_sensitivity,
            -mouse_dx.to_radians() as f32 * self.rotate_sensitivity,
        );
    }

    fn update_camera(&mut self, camera: &mut Camera, dt: std::time::Duration) {
        let dt = dt.as_secs_f32();

        let translation = na::Vector3::new(
            self.translation.x * dt,
            self.translation.y * dt,
            self.translation.z * dt,
        );
        let (r, p) = self.rotation;
        // let (curr_r, curr_p, curr_y) = camera.cam_to_world.rotation.euler_angles();
        // let next_r = curr_r + r * dt;
        // let next_p = curr_p + p * dt;
        // let next_y = curr_y + self.spin * dt;
        if r != 0.0 || p != 0.0 || self.spin != 0.0 {
            let rotation = na::UnitQuaternion::from_euler_angles(r * dt, p * dt, self.spin * dt);
            camera.cam_to_world.rotation *= rotation;
        }

        let translation = camera.cam_to_world.transform_vector(&translation);
        let translation = na::Translation3::from(translation);
        camera.cam_to_world.append_translation_mut(&translation);

        trace!(
            self.log,
            "camera is now at: {:?}, rotation: {:?}",
            camera.cam_to_world.translation,
            camera.cam_to_world.rotation.euler_angles()
        );

        self.translation = na::Translation3::identity();
        self.rotation = (0.0, 0.0);
        self.spin = 0.0;
    }

    fn require_mouse_press(&self) -> bool {
        true
    }
}
