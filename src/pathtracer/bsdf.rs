use super::{
    bxdf::{BxDF, BxDFInterface, BxDFType},
    interaction::SurfaceMediumInteraction,
};
use crate::common::spectrum::Spectrum;

const MAX_BXDFS: usize = 8;
pub struct BSDF {
    pub eta: f32,
    ns: na::Vector3<f32>,
    ng: na::Vector3<f32>,
    ss: na::Vector3<f32>,
    ts: na::Vector3<f32>,
    n_bxdfs: usize,
    bxdfs: [Option<BxDF>; MAX_BXDFS],
    log: slog::Logger,
}

impl BSDF {
    pub fn new(log: &slog::Logger, si: &SurfaceMediumInteraction, eta: f32) -> Self {
        let log = log.new(o!());
        let ns = si.shading.n;
        let ss = si.shading.dpdu.normalize();
        Self {
            eta,
            ns,
            ng: si.general.n,
            ss,
            ts: ns.cross(&ss),
            n_bxdfs: 0,
            bxdfs: Default::default(),
            log,
        }
    }

    pub fn num_components(&self, flags: BxDFType) -> usize {
        let mut num = 0;

        for i in 0..self.n_bxdfs {
            if self.bxdfs[i].as_ref().unwrap().matches_flags(flags) {
                num += 1;
            }
        }

        num
    }

    pub fn add(&mut self, b: BxDF) {
        debug_assert!(self.n_bxdfs < MAX_BXDFS);
        self.bxdfs[self.n_bxdfs] = Some(b);
        self.n_bxdfs += 1;
    }

    pub fn world_to_local(&self, v: &na::Vector3<f32>) -> na::Vector3<f32> {
        na::Vector3::new(v.dot(&self.ss), v.dot(&self.ts), v.dot(&self.ns))
    }

    pub fn local_to_world(&self, v: &na::Vector3<f32>) -> na::Vector3<f32> {
        na::Vector3::new(
            self.ss.x * v.x + self.ts.x * v.y + self.ns.x * v.z,
            self.ss.y * v.x + self.ts.y * v.y + self.ns.y * v.z,
            self.ss.z * v.x + self.ts.z * v.y + self.ns.z * v.z,
        )
    }

    pub fn sample_f(
        &self,
        wo_world: &na::Vector3<f32>,
        wi_world: &mut na::Vector3<f32>,
        u: &na::Point2<f32>,
        pdf: &mut f32,
        bxdf_type: BxDFType,
        sampled_type: &mut Option<BxDFType>,
    ) -> Spectrum {
        let matching_comps = self.num_components(bxdf_type);
        if matching_comps == 0 {
            *pdf = 0.0;
            if let Some(sampled_type) = sampled_type {
                *sampled_type = BxDFType::empty();
            }
            return Spectrum::new(0.0);
        }
        let comp = ((u[0] * matching_comps as f32).floor() as usize).min(matching_comps - 1);
        let mut bxdf = None;
        let mut count = comp;
        for i in 0..self.n_bxdfs {
            if self.bxdfs[i].as_ref().unwrap().matches_flags(bxdf_type) {
                if count == 0 {
                    bxdf = self.bxdfs[i].as_ref();
                    break;
                }
                count -= 1;
            }
        }
        let bxdf = bxdf.unwrap();

        let u_remapped = na::Point2::new((u[0] * matching_comps as f32) - comp as f32, u[1]);
        let mut wi = glm::zero();
        let wo = self.world_to_local(wo_world);
        *pdf = 0.0;

        if let Some(sampled_type) = sampled_type {
            *sampled_type = bxdf.get_type();
        }

        let mut f = bxdf.sample_f(&wo, &mut wi, &u_remapped, pdf, sampled_type);

        if *pdf == 0.0 {
            if let Some(sampled_type) = sampled_type {
                *sampled_type = BxDFType::empty();
            }
            return Spectrum::new(0.0);
        }

        *wi_world = self.local_to_world(&wi);

        if !(bxdf.get_type().contains(BxDFType::BSDF_SPECULAR)) && matching_comps > 1 {
            for i in 0..self.n_bxdfs {
                let curr_bxdf = self.bxdfs[i].as_ref().unwrap();
                if curr_bxdf as *const _ != bxdf as *const _ && curr_bxdf.matches_flags(bxdf_type) {
                    *pdf += curr_bxdf.pdf(&wo, &wi);
                }
            }
        }
        if matching_comps > 1 {
            *pdf /= matching_comps as f32;
        }

        if !(bxdf.get_type().contains(BxDFType::BSDF_SPECULAR)) && matching_comps > 1 {
            let reflect = wi_world.dot(&self.ng) * wo_world.dot(&self.ng) > 0.0;
            f = Spectrum::new(0.0);
            for i in 0..self.n_bxdfs {
                if let Some(ref curr_bxdf) = self.bxdfs[i] {
                    if curr_bxdf.matches_flags(bxdf_type)
                        && ((reflect && (curr_bxdf.get_type().contains(BxDFType::BSDF_REFLECTION)))
                            || (!reflect
                                && (curr_bxdf.get_type().contains(BxDFType::BSDF_TRANSMISSION))))
                    {
                        f += curr_bxdf.f(&wo, &wi);
                    }
                } else {
                    panic!("bxdf does not exist in index {:?}", i);
                }
            }
        }

        f
    }

    pub fn f(&self, wo_w: &na::Vector3<f32>, wi_w: &na::Vector3<f32>, flags: BxDFType) -> Spectrum {
        let wi = self.world_to_local(&wi_w);
        let wo = self.world_to_local(&wo_w);
        if wo.z == 0.0 {
            return Spectrum::new(0.0);
        }
        let reflect = wi_w.dot(&self.ng) * wo_w.dot(&self.ng) > 0.0;

        trace!(self.log, "local wi: {:?}, local wo: {:?}", wi, wo);
        trace!(self.log, "ng: {:?}, ns: {:?}", self.ng, self.ns);
        trace!(self.log, "f"; "w ng dots" => slog::FnValue(|_ : &slog::Record|
            format!("wi_w dot ng: {:?}, wo_w dot ng: {:?}",
            wi_w.dot(&self.ng),
            wo_w.dot(&self.ng))
        ));

        trace!(self.log, "f"; "w ns dots" => slog::FnValue(|_ : &slog::Record|
            format!("wi_w dot ns: {:?}, wo_w dot ns: {:?}",
            wi_w.dot(&self.ns),
            wo_w.dot(&self.ns))
        ));
        let mut f = Spectrum::new(0.0);

        for i in 0..self.n_bxdfs {
            if let Some(ref bxdf) = self.bxdfs[i] {
                if bxdf.matches_flags(flags)
                    && ((reflect && (bxdf.get_type().contains(BxDFType::BSDF_REFLECTION)))
                        || (!reflect && (bxdf.get_type().contains(BxDFType::BSDF_TRANSMISSION))))
                {
                    f += bxdf.f(&wo, &wi);
                }
            } else {
                panic!("bxdf does not exist in index {:?}", i);
            }
        }

        f
    }

    pub fn pdf(
        &self,
        wo_world: &na::Vector3<f32>,
        wi_world: &na::Vector3<f32>,
        flags: BxDFType,
    ) -> f32 {
        if self.n_bxdfs == 0 {
            return 0.0;
        }

        let wo = self.world_to_local(&wo_world);
        let wi = self.world_to_local(&wi_world);

        if wo.z == 0.0 {
            return 0.0;
        }

        let mut pdf = 0.0;
        let mut matching_comps = 0;

        for i in 0..self.n_bxdfs {
            let bxdf = self.bxdfs[i].as_ref().unwrap();
            if bxdf.matches_flags(flags) {
                matching_comps += 1;
                pdf += bxdf.pdf(&wo, &wi);
            }
        }

        if matching_comps > 0 {
            pdf / matching_comps as f32
        } else {
            0.0
        }
    }
}
