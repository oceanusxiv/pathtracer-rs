use super::{bsdf::BSDF, primitive::Primitive, shape::Shape, TransportMode};
use crate::common::{
    math::{face_forward, offset_ray_origin, solve_linear_system_2x2},
    ray::{Ray, RayDifferential},
    spectrum::Spectrum,
};
use std::cell::RefCell;

#[derive(Clone, Copy, Debug)]
pub struct Interaction {
    pub p: na::Point3<f32>,
    pub time: f32,
    pub p_error: na::Vector3<f32>,
    pub wo: na::Vector3<f32>,
    pub n: na::Vector3<f32>,
}

impl Default for Interaction {
    fn default() -> Self {
        Interaction {
            p: na::Point3::origin(),
            time: 0.0,
            p_error: glm::zero(),
            wo: glm::zero(),
            n: glm::zero(),
        }
    }
}

const SHADOW_EPSILON: f32 = 0.0001;

impl Interaction {
    pub fn spawn_ray(&self, d: &na::Vector3<f32>) -> Ray {
        let o = offset_ray_origin(&self.p, &self.p_error, &self.n, d);
        Ray {
            o,
            d: *d,
            t_max: RefCell::new(f32::INFINITY),
        }
    }
    pub fn spawn_ray_to(&self, p2: &na::Point3<f32>) -> Ray {
        let origin = offset_ray_origin(&self.p, &self.p_error, &self.n, &(p2 - self.p));
        let d = p2 - origin;
        Ray {
            o: origin,
            d: d,
            t_max: RefCell::new(1.0 - SHADOW_EPSILON),
        }
    }

    pub fn spawn_ray_to_it(&self, it2: &Interaction) -> Ray {
        let origin = offset_ray_origin(&self.p, &self.p_error, &self.n, &(it2.p - self.p));
        let target = offset_ray_origin(&it2.p, &it2.p_error, &it2.n, &(origin - it2.p));
        let d = target - origin;
        return Ray {
            o: origin,
            d,
            t_max: RefCell::new(1.0 - SHADOW_EPSILON),
        };
    }
}

pub struct SurfaceInteractionShading {
    pub n: na::Vector3<f32>,
    pub dpdu: na::Vector3<f32>,
    pub dpdv: na::Vector3<f32>,
    pub dndu: na::Vector3<f32>,
    pub dndv: na::Vector3<f32>,
}

impl Default for SurfaceInteractionShading {
    fn default() -> Self {
        SurfaceInteractionShading {
            n: glm::zero(),
            dpdu: glm::zero(),
            dpdv: glm::zero(),
            dndu: glm::zero(),
            dndv: glm::zero(),
        }
    }
}

pub struct SurfaceMediumInteraction<'a> {
    pub general: Interaction,
    pub uv: na::Point2<f32>,
    pub dpdu: na::Vector3<f32>,
    pub dpdv: na::Vector3<f32>,
    pub dndu: na::Vector3<f32>,
    pub dndv: na::Vector3<f32>,
    pub shading: SurfaceInteractionShading,
    pub shape: Option<&'a dyn Shape>,
    pub primitive: Option<&'a dyn Primitive>,
    pub bsdf: Option<BSDF>,

    pub dpdx: na::Vector3<f32>,
    pub dpdy: na::Vector3<f32>,
    pub dudx: f32,
    pub dvdx: f32,
    pub dudy: f32,
    pub dvdy: f32,
}

impl<'a> Default for SurfaceMediumInteraction<'a> {
    fn default() -> SurfaceMediumInteraction<'a> {
        SurfaceMediumInteraction {
            general: Interaction {
                p: na::Point3::origin(),
                time: 0.0,
                p_error: glm::zero(),
                wo: glm::zero(),
                n: glm::zero(),
            },
            uv: na::Point2::new(0.0, 0.0),
            dpdu: glm::zero(),
            dpdv: glm::zero(),
            dndu: glm::zero(),
            dndv: glm::zero(),
            shading: Default::default(),
            shape: None,
            primitive: None,
            bsdf: None,
            dpdx: glm::zero(),
            dpdy: glm::zero(),
            dudx: 0.0,
            dvdx: 0.0,
            dudy: 0.0,
            dvdy: 0.0,
        }
    }
}

impl<'a> SurfaceMediumInteraction<'a> {
    pub fn new(
        p: &na::Point3<f32>,
        p_error: &na::Vector3<f32>,
        uv: &na::Point2<f32>,
        wo: &na::Vector3<f32>,
        dpdu: &na::Vector3<f32>,
        dpdv: &na::Vector3<f32>,
        dndu: &na::Vector3<f32>,
        dndv: &na::Vector3<f32>,
        time: f32,
        shape: &'a dyn Shape,
    ) -> Self {
        let n = dpdu.cross(dpdv).normalize();
        let shading = SurfaceInteractionShading {
            n,
            dpdu: *dpdu,
            dpdv: *dpdv,
            dndu: *dndu,
            dndv: *dndv,
        };
        Self {
            general: Interaction {
                p: *p,
                time,
                p_error: *p_error,
                wo: *wo,
                n,
            },
            uv: *uv,
            dpdu: *dpdu,
            dpdv: *dpdv,
            dndu: *dndu,
            dndv: *dndv,
            shading,
            shape: Some(shape),
            primitive: None,
            bsdf: None,
            ..Default::default()
        }
    }

    pub fn set_shading_geometry(
        &mut self,
        dpdus: &na::Vector3<f32>,
        dpdvs: &na::Vector3<f32>,
        dndus: &na::Vector3<f32>,
        dndvs: &na::Vector3<f32>,
        orientation_is_authoritative: bool,
    ) {
        // Compute _shading.n_ for _SurfaceInteraction_
        self.shading.n = dpdus.cross(dpdvs).normalize();
        if orientation_is_authoritative {
            self.general.n = face_forward(&self.general.n, &self.shading.n);
        } else {
            self.shading.n = face_forward(&self.shading.n, &self.general.n);
        }

        // Initialize _shading_ partial derivative values
        self.shading.dpdu = *dpdus;
        self.shading.dpdv = *dpdvs;
        self.shading.dndu = *dndus;
        self.shading.dndv = *dndvs;
    }

    pub fn compute_differentials(&mut self, ray: &RayDifferential) -> bool {
        if ray.has_differentials {
            let n = &self.general.n;
            let p = &self.general.p;
            // Estimate screen space change in $\pt{}$ and $(u,v)$

            // Compute auxiliary intersection points with plane
            let d = n.dot(&p.coords);
            let tx = -(n.dot(&ray.rx_origin.coords) - d) / n.dot(&ray.rx_direction);
            if tx.is_infinite() || tx.is_nan() {
                return false;
            }
            let px = ray.rx_origin + tx * ray.rx_direction;
            let ty = -(n.dot(&ray.ry_origin.coords) - d) / n.dot(&ray.ry_direction);
            if ty.is_infinite() || ty.is_nan() {
                return false;
            }
            let py = ray.ry_origin + ty * ray.ry_direction;
            self.dpdx = px - p;
            self.dpdy = py - p;

            // Compute $(u,v)$ offsets at auxiliary points

            // Choose two dimensions to use for ray offset computation
            let mut dim = [0, 0];
            if n.x.abs() > n.y.abs() && n.x.abs() > n.y.abs() {
                dim[0] = 1;
                dim[1] = 2;
            } else if n.y.abs() > n.z.abs() {
                dim[0] = 0;
                dim[1] = 2;
            } else {
                dim[0] = 0;
                dim[1] = 1;
            }

            let a = na::Matrix2::new(
                self.dpdu[dim[0]],
                self.dpdv[dim[0]],
                self.dpdu[dim[1]],
                self.dpdv[dim[1]],
            );
            let bx = na::Vector2::new(px[dim[0]] - p[dim[0]], px[dim[1]] - p[dim[1]]);
            let by = na::Vector2::new(py[dim[0]] - p[dim[0]], py[dim[1]] - p[dim[1]]);

            if let Some(result) = solve_linear_system_2x2(&a, &bx) {
                self.dudx = result[0];
                self.dvdx = result[1];
            } else {
                self.dudx = 0.0;
                self.dvdx = 0.0;
            }

            if let Some(result) = solve_linear_system_2x2(&a, &by) {
                self.dudy = result[0];
                self.dvdy = result[1];
            } else {
                self.dudy = 0.0;
                self.dvdy = 0.0;
            }

            return true;
        } else {
            return false;
        }
    }

    pub fn compute_scattering_functions(&mut self, r: &RayDifferential, mode: TransportMode) {
        if !self.compute_differentials(&r) {
            self.dudx = 0.0;
            self.dvdx = 0.0;
            self.dudy = 0.0;
            self.dvdy = 0.0;
            self.dpdx = glm::zero();
            self.dpdy = glm::zero();
        }
        self.primitive
            .unwrap()
            .compute_scattering_functions(self, mode);
    }

    pub fn le(&self, w: &na::Vector3<f32>) -> Spectrum {
        if let Some(area) = self.primitive.unwrap().get_area_light() {
            area.L(&self, &w)
        } else {
            Spectrum::new(0.0)
        }
    }

    pub fn is_surface_interaction(&self) -> bool {
        self.shading.n != na::Vector3::zeros()
    }
}
