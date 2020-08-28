use super::SurfaceInteraction;
use crate::common::bounds::Bounds3;
use crate::common::math::*;
use crate::common::ray::Ray;
use crate::common::{Mesh, Object};
use std::sync::Arc;

pub trait Shape {
    fn intersect(&self, r: &Ray, t_hit: &mut f32, isect: &mut SurfaceInteraction) -> bool;
    fn world_bound(&self) -> Bounds3;
}

pub struct Triangle {
    mesh: Arc<Mesh>,
    indices: [u32; 3],
}

impl Triangle {}

impl Shape for Triangle {
    fn intersect(&self, r: &Ray, t_hit: &mut f32, isect: &mut SurfaceInteraction) -> bool {
        // get triangle vertices
        let p0 = self.mesh.pos[self.indices[0] as usize];
        let p1 = self.mesh.pos[self.indices[1] as usize];
        let p2 = self.mesh.pos[self.indices[2] as usize];

        // perform ray-triangle intersection test

        // transform triangle vertices to ray coordinate space

        // translate vertices based on ray origin
        let mut p0t = p0 - r.o;
        let mut p1t = p1 - r.o;
        let mut p2t = p2 - r.o;
        // permute components of triangle vertices and ray direction
        let kz = max_dimension(&glm::abs(&r.d));
        let mut kx = kz + 1;
        if kx == 3 {
            kx = 0;
        }
        let mut ky = kx + 1;
        if ky == 3 {
            ky = 0;
        }
        let d = permute(&r.d, kx, ky, kz);
        p0t = permute(&p0t, kx, ky, kz);
        p1t = permute(&p1t, kx, ky, kz);
        p2t = permute(&p2t, kx, ky, kz);
        // apply shear transformation to translated vertex positions
        let sx = -d.x / d.z;
        let sy = -d.y / d.z;
        let sz = 1.0f32 / d.z;

        p0t.x += sx * p0t.z;
        p0t.y += sy * p0t.z;
        p1t.x += sx * p1t.z;
        p1t.y += sy * p1t.z;
        p2t.x += sx * p2t.z;
        p2t.y += sy * p2t.z;

        // compute edge function coefficients e0, e1, and e2
        let mut e0 = p1t.x * p2t.y - p1t.y * p2t.x;
        let mut e1 = p2t.x * p0t.y - p2t.y * p0t.x;
        let mut e2 = p0t.x * p1t.y - p0t.y * p1t.x;

        if e0 == 0.0 || e1 == 0.0 || e2 == 0.0 {
            let p2txp1ty = p2t.x as f64 * p1t.y as f64;
            let p2typ1tx = p2t.y as f64 * p1t.x as f64;
            e0 = (p2typ1tx - p2txp1ty) as f32;
            let p0txp2ty = p0t.x as f64 * p2t.y as f64;
            let p0typ2tx = p0t.y as f64 * p2t.x as f64;
            e1 = (p0typ2tx - p0txp2ty) as f32;
            let p1txp0ty = p1t.x as f64 * p0t.y as f64;
            let p1typ0tx = p1t.y as f64 * p0t.x as f64;
            e2 = (p1typ0tx - p1txp0ty) as f32;
        }

        // Perform triangle edge and determinant tests
        if (e0 < 0.0 || e1 < 0.0 || e2 < 0.0) && (e0 > 0.0 || e1 > 0.0 || e2 > 0.0) {
            return false;
        }
        let det = e0 + e1 + e2;
        if det == 0.0 {
            return false;
        }

        // Compute scaled hit distance to triangle and test against ray t range
        p0t.z *= sz;
        p1t.z *= sz;
        p2t.z *= sz;
        let t_scaled = e0 * p0t.z + e1 * p1t.z + e2 * p2t.z;
        if det < 0.0 && (t_scaled >= 0.0 || t_scaled < *r.t_max.borrow() * det) {
            return false;
        } else if det > 0.0 && (t_scaled <= 0.0 || t_scaled > *r.t_max.borrow() * det) {
            return false;
        }

        // Compute barycentric coordinates and t value for triangle intersection
        let inv_det = 1.0 / det;
        let b0 = e0 * inv_det;
        let b1 = e1 * inv_det;
        let b2 = e2 * inv_det;
        let t = t_scaled * inv_det;

        // ensure that computed triangle t is conservatively greater than zero

        // compute delta_z term for triangle t error bounds
        let max_z_t = glm::comp_max(&glm::abs(&glm::vec3(p0t.z, p1t.z, p2t.z)));
        let delta_z = gamma(3) * max_z_t;

        // compute delta_x and delta_y terms for triangle t error bounds
        let max_x_t = glm::comp_max(&glm::abs(&glm::vec3(p0t.x, p1t.x, p2t.x)));
        let max_y_t = glm::comp_max(&glm::abs(&glm::vec3(p0t.y, p1t.y, p2t.y)));
        let delta_x = gamma(5) * (max_x_t + max_z_t);
        let delta_y = gamma(5) * (max_y_t + max_z_t);

        // compute delta_e term for triangle t error bounds
        let delta_e = 2.0 * (gamma(2) * max_x_t * max_y_t + delta_y * max_x_t + delta_x * max_y_t);

        // compute delta_t term for triangle t error bounds and check t
        let max_e = glm::comp_max(&glm::abs(&glm::vec3(e0, e1, e2)));
        let delta_t = 3.0
            * (gamma(3) * max_e * max_z_t + delta_e * max_z_t + delta_z * max_e)
            * inv_det.abs();
        if t <= delta_t {
            return false;
        }

        // Compute triangle partial derivatives

        // Compute deltas for triangle partial derivatives
        let dp02 = p0 - p2;
        let dp12 = p1 - p2;

        // Compute error bounds for triangle intersection
        let x_abs_sum = (b0 * p0.x).abs() + (b1 * p1.x).abs() + (b2 * p2.x).abs();
        let y_abs_sum = (b0 * p0.y).abs() + (b1 * p1.y).abs() + (b2 * p2.y).abs();
        let z_abs_sum = (b0 * p0.z).abs() + (b1 * p1.z).abs() + (b2 * p2.z).abs();
        let p_error: glm::Vec3 = gamma(7) * glm::vec3(x_abs_sum, y_abs_sum, z_abs_sum);

        // Interpolate (u,v) parametric coordinates and hit point
        let p_hit = b0 * p0.coords + b1 * p1.coords + b2 * p2.coords;

        *t_hit = t;
        (*isect).p = p_hit;
        (*isect).p_error = p_error;
        (*isect).wo = -r.d;
        (*isect).n = glm::normalize(&glm::cross(&dp02, &dp12));

        return true;
    }

    fn world_bound(&self) -> Bounds3 {
        let p0 = self.mesh.pos[self.indices[0] as usize];
        let p1 = self.mesh.pos[self.indices[1] as usize];
        let p2 = self.mesh.pos[self.indices[2] as usize];
        Bounds3::union_p(&Bounds3::new(p0, p1), &p2)
    }
}

impl Mesh {
    pub fn to_shapes(&self, object: &Object) -> Vec<Box<dyn Shape + Send + Sync>> {
        let mut world_mesh = (*self).clone();

        for pos in &mut world_mesh.pos {
            *pos = object.obj_to_world * *pos;
        }

        let mut shapes = Vec::new();

        let world_mesh = Arc::new(world_mesh);
        for chunk in world_mesh.indices.chunks_exact(3) {
            shapes.push(Box::new(Triangle {
                mesh: Arc::clone(&world_mesh),
                indices: [chunk[0], chunk[1], chunk[2]],
            }) as Box<dyn Shape + Send + Sync>)
        }

        shapes
    }
}
