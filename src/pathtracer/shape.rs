use super::{
    texture::{ImageTexture, SyncTexture, UVMap},
    SurfaceInteraction,
};
use crate::common::bounds::Bounds3;
use crate::common::math::*;
use crate::common::ray::Ray;
use crate::common::{Mesh, Object, TextureInfo};
use std::sync::Arc;

pub trait Shape {
    fn intersect<'a>(
        &'a self,
        r: &Ray,
        t_hit: &mut f32,
        isect: &mut SurfaceInteraction<'a>,
    ) -> bool;
    fn intersect_p(&self, r: &Ray) -> bool;
    fn world_bound(&self) -> Bounds3;
}

pub trait SyncShape: Shape + Send + Sync {}
impl<T> SyncShape for T where T: Shape + Send + Sync {}

pub struct Triangle {
    mesh: Arc<TriangleMesh>,
    indices: [u32; 3],
    reverse_orientation: bool,
    transform_swaps_handedness: bool,
}

impl Triangle {
    fn get_uvs(&self) -> [na::Point2<f32>; 3] {
        if !self.mesh.uv.is_empty() {
            [
                self.mesh.uv[self.indices[0] as usize],
                self.mesh.uv[self.indices[1] as usize],
                self.mesh.uv[self.indices[2] as usize],
            ]
        } else {
            [
                na::Point2::new(0.0, 0.0),
                na::Point2::new(1.0, 0.0),
                na::Point2::new(1.0, 1.0),
            ]
        }
    }
}

impl Shape for Triangle {
    fn intersect<'a>(
        &'a self,
        r: &Ray,
        t_hit: &mut f32,
        isect: &mut SurfaceInteraction<'a>,
    ) -> bool {
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
        let mut dpdu = na::Vector3::new(0.0, 0.0, 0.0);
        let mut dpdv = na::Vector3::new(0.0, 0.0, 0.0);
        let uv = self.get_uvs();

        // Compute deltas for triangle partial derivatives
        let duv02 = uv[0] - uv[2];
        let duv12 = uv[1] - uv[2];
        let dp02 = p0 - p2;
        let dp12 = p1 - p2;
        let determinant = duv02[0] * duv12[1] - duv02[1] * duv12[0];
        let degenerate_uv = determinant.abs() < 1e-8;

        if !degenerate_uv {
            let invdet = 1.0 / determinant;
            dpdu = (duv12[1] * dp02 - duv02[1] * dp12) * invdet;
            dpdv = (-duv12[0] * dp02 + duv02[0] * dp12) * invdet;
        }
        if degenerate_uv || dpdu.cross(&dpdv).norm_squared() == 0.0 {
            // Handle zero determinant for triangle partial derivative matrix
            let ng = (p2 - p0).cross(&(p1 - p0));
            if ng.norm_squared() == 0.0 {
                // The triangle is actually degenerate; the intersection is
                // bogus.
                return false;
            }

            coordinate_system(&ng.normalize(), &mut dpdu, &mut dpdv);
        }

        // Compute error bounds for triangle intersection
        let x_abs_sum = (b0 * p0.x).abs() + (b1 * p1.x).abs() + (b2 * p2.x).abs();
        let y_abs_sum = (b0 * p0.y).abs() + (b1 * p1.y).abs() + (b2 * p2.y).abs();
        let z_abs_sum = (b0 * p0.z).abs() + (b1 * p1.z).abs() + (b2 * p2.z).abs();
        let p_error: glm::Vec3 = gamma(7) * glm::vec3(x_abs_sum, y_abs_sum, z_abs_sum);

        // Interpolate (u,v) parametric coordinates and hit point
        let p_hit = b0 * p0.coords + b1 * p1.coords + b2 * p2.coords;
        let uv_hit = b0 * uv[0].coords + b1 * uv[1].coords + b2 * uv[2].coords;

        // Test intersection against alpha texture, if present
        if let Some(alpha_mask) = self.mesh.alpha_mask.as_ref() {
            let isect_local = SurfaceInteraction::new(
                &na::Point3::from(p_hit),
                &glm::zero(),
                &na::Point2::from(uv_hit),
                &-r.d,
                &dpdu,
                &dpdv,
                &glm::zero(),
                &glm::zero(),
                0.0,
                self,
            );
            if alpha_mask.evaluate(&isect_local) == 0.0 {
                return false;
            }
        }

        // Fill in SurfaceInteraction from triangle hit
        (*isect) = SurfaceInteraction::new(
            &na::Point3::from(p_hit),
            &p_error,
            &na::Point2::from(uv_hit),
            &-r.d,
            &dpdu,
            &dpdv,
            &glm::zero(),
            &glm::zero(),
            0.0,
            self,
        );

        // Override surface normal in isect for triangle
        (*isect).general.n = glm::normalize(&glm::cross(&dp02, &dp12));
        (*isect).shading.n = (*isect).general.n;

        if !self.mesh.normal.is_empty() || !self.mesh.s.is_empty() {
            // Initialize _Triangle_ shading geometry

            // Compute shading normal _ns_ for triangle
            let mut ns;
            if !self.mesh.normal.is_empty() {
                let n0 = self.mesh.normal[self.indices[0] as usize];
                let n1 = self.mesh.normal[self.indices[1] as usize];
                let n2 = self.mesh.normal[self.indices[2] as usize];
                ns = b0 * n0 + b1 * n1 + b2 * n2;
                if ns.norm_squared() > 0.0 {
                    ns = ns.normalize();
                } else {
                    ns = isect.general.n;
                }
            } else {
                ns = isect.general.n
            }

            // Compute shading tangent _ss_ for triangle
            let mut ss;
            if !self.mesh.s.is_empty() {
                let s0 = self.mesh.s[self.indices[0] as usize];
                let s1 = self.mesh.s[self.indices[1] as usize];
                let s2 = self.mesh.s[self.indices[2] as usize];

                ss = b0 * s0 + b1 * s1 + b2 * s2;
                if ss.norm_squared() > 0.0 {
                    ss = ss.normalize();
                } else {
                    ss = isect.dpdu.normalize();
                }
            } else {
                ss = isect.dpdu.normalize()
            }

            // Compute shading bitangent _ts_ for triangle and adjust _ss_
            let mut ts = ss.cross(&ns);
            if ts.norm_squared() > 0.0 {
                ts = ts.normalize();
                ss = ts.cross(&ns);
            } else {
                coordinate_system(&ns, &mut ss, &mut ts);
            }

            // Compute dndu and dndv for triangle shading geometry
            let mut dndu = na::Vector3::new(0.0, 0.0, 0.0);
            let mut dndv = na::Vector3::new(0.0, 0.0, 0.0);
            if !self.mesh.normal.is_empty() {
                // Compute deltas for triangle partial derivatives of normal
                let duv02 = uv[0] - uv[2];
                let duv12 = uv[1] - uv[2];
                let n0 = self.mesh.normal[self.indices[0] as usize];
                let n1 = self.mesh.normal[self.indices[1] as usize];
                let n2 = self.mesh.normal[self.indices[2] as usize];
                let dn1 = n0 - n2;
                let dn2 = n1 - n2;
                let determinant = duv02[0] * duv12[1] - duv02[1] * duv12[0];
                let degenerate_uv = determinant.abs() < 1e-8;

                if degenerate_uv {
                    // We can still compute dndu and dndv, with respect to the
                    // same arbitrary coordinate system we use to compute dpdu
                    // and dpdv when this happens. It's important to do this
                    // (rather than giving up) so that ray differentials for
                    // rays reflected from triangles with degenerate
                    // parameterizations are still reasonable.
                    let dn = (n2 - n0).cross(&(n1 - n0));
                    if dn.norm_squared() == 0.0 {
                        dndu = na::Vector3::new(0.0, 0.0, 0.0);
                        dndv = na::Vector3::new(0.0, 0.0, 0.0);
                    } else {
                        let mut dnu = na::Vector3::new(0.0, 0.0, 0.0);
                        let mut dnv = na::Vector3::new(0.0, 0.0, 0.0);
                        coordinate_system(&dn, &mut dnu, &mut dnv);
                        dndu = dnu;
                        dndv = dnv;
                    }
                } else {
                    let inv_det = 1.0 / determinant;
                    dndu = (duv12[1] * dn1 - duv02[1] * dn2) * inv_det;
                    dndv = (-duv12[0] * dn1 + duv02[0] * dn2) * inv_det;
                }
            }
            if self.reverse_orientation {
                ts = -ts;
            }
            isect.set_shading_geometry(&ss, &ts, &dndu, &dndv, true);
        }

        *t_hit = t;
        return true;
    }

    fn intersect_p(&self, r: &Ray) -> bool {
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

        // Test shadow ray intersection against alpha texture, if present
        if let Some(alpha_mask) = self.mesh.alpha_mask.as_ref() {
            // Compute triangle partial derivatives
            let mut dpdu = na::Vector3::new(0.0, 0.0, 0.0);
            let mut dpdv = na::Vector3::new(0.0, 0.0, 0.0);
            let uv = self.get_uvs();

            // Compute deltas for triangle partial derivatives
            let duv02 = uv[0] - uv[2];
            let duv12 = uv[1] - uv[2];
            let dp02 = p0 - p2;
            let dp12 = p1 - p2;
            let determinant = duv02[0] * duv12[1] - duv02[1] * duv12[0];
            let degenerate_uv = determinant.abs() < 1e-8;

            if !degenerate_uv {
                let invdet = 1.0 / determinant;
                dpdu = (duv12[1] * dp02 - duv02[1] * dp12) * invdet;
                dpdv = (-duv12[0] * dp02 + duv02[0] * dp12) * invdet;
            }
            if degenerate_uv || dpdu.cross(&dpdv).norm_squared() == 0.0 {
                // Handle zero determinant for triangle partial derivative matrix
                let ng = (p2 - p0).cross(&(p1 - p0));
                if ng.norm_squared() == 0.0 {
                    // The triangle is actually degenerate; the intersection is
                    // bogus.
                    return false;
                }

                coordinate_system(&ng.normalize(), &mut dpdu, &mut dpdv);
            }

            // Interpolate (u,v) parametric coordinates and hit point
            let p_hit = b0 * p0.coords + b1 * p1.coords + b2 * p2.coords;
            let uv_hit = b0 * uv[0].coords + b1 * uv[1].coords + b2 * uv[2].coords;

            let isect_local = SurfaceInteraction::new(
                &na::Point3::from(p_hit),
                &glm::zero(),
                &na::Point2::from(uv_hit),
                &-r.d,
                &dpdu,
                &dpdv,
                &glm::zero(),
                &glm::zero(),
                0.0,
                self,
            );
            if alpha_mask.evaluate(&isect_local) == 0.0 {
                return false;
            }
        }

        return true;
    }

    fn world_bound(&self) -> Bounds3 {
        let p0 = self.mesh.pos[self.indices[0] as usize];
        let p1 = self.mesh.pos[self.indices[1] as usize];
        let p2 = self.mesh.pos[self.indices[2] as usize];
        Bounds3::union_p(&Bounds3::new(p0, p1), &p2)
    }
}

pub struct TriangleMesh {
    pub indices: Vec<u32>,
    pub pos: Vec<na::Point3<f32>>,
    pub normal: Vec<na::Vector3<f32>>,
    pub s: Vec<na::Vector3<f32>>,
    pub uv: Vec<na::Point2<f32>>,
    pub colors: Vec<na::Vector3<f32>>,
    pub alpha_mask: Option<Arc<dyn SyncTexture<f32>>>,
}

pub fn shape_from_mesh(
    log: &slog::Logger,
    mesh: &Mesh,
    object: &Object,
    alpha_mask: Option<&TextureInfo<image::GrayImage>>,
) -> Vec<Box<dyn SyncShape>> {
    let mut alpha_mask_texture = None;
    if let Some(alpha_mask_info) = alpha_mask {
        alpha_mask_texture = Some(Arc::new(ImageTexture::<f32>::new(
            log,
            &alpha_mask_info.image,
            1.0,
            alpha_mask_info.sampler_info.wrap_mode,
            UVMap::new(
                alpha_mask_info.image.width() as f32,
                alpha_mask_info.image.height() as f32,
                0.0,
                0.0,
            ),
        )) as Arc<dyn SyncTexture<f32>>);
    }
    let mut world_mesh = TriangleMesh {
        indices: mesh.indices.clone(),
        pos: mesh.pos.clone(),
        normal: mesh.normal.clone(),
        s: mesh.s.clone(),
        uv: mesh.uv.clone(),
        colors: mesh.colors.clone(),
        alpha_mask: alpha_mask_texture,
    };

    for pos in &mut world_mesh.pos {
        *pos = object.obj_to_world * *pos;
    }

    for normal in &mut world_mesh.normal {
        *normal = object.obj_to_world * *normal;
    }

    for s in &mut world_mesh.s {
        *s = object.obj_to_world * *s;
    }

    let mut shapes = Vec::new();

    let world_mesh = Arc::new(world_mesh);
    for chunk in world_mesh.indices.chunks_exact(3) {
        shapes.push(Box::new(Triangle {
            mesh: Arc::clone(&world_mesh),
            indices: [chunk[0], chunk[1], chunk[2]],
            reverse_orientation: false,
            transform_swaps_handedness: false,
        }) as Box<dyn SyncShape>)
    }

    shapes
}
