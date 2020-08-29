use super::sampling::cosine_sample_hemisphere;
use super::{SurfaceInteraction, TransportMode};
use crate::common::spectrum::Spectrum;

bitflags! {
    pub struct BxDFType: u32 {
        const BSDF_REFLECTION = 1 << 0;
        const BSDF_TRANSMISSION = 1 << 1;
        const BSDF_DIFFUSE = 1 << 2;
        const BSDF_GLOSSY = 1 << 3;
        const BSDF_SPECULAR = 1 << 4;
        const BSDF_ALL = Self::BSDF_DIFFUSE.bits | Self::BSDF_GLOSSY.bits | Self::BSDF_SPECULAR.bits | Self::BSDF_REFLECTION.bits |
        Self::BSDF_TRANSMISSION.bits;
    }
}

fn cos_theta(w: &na::Vector3<f32>) -> f32 {
    w.z
}

fn cos_2_theta(w: &na::Vector3<f32>) -> f32 {
    w.z * w.z
}

fn abs_cos_theta(w: &na::Vector3<f32>) -> f32 {
    w.z.abs()
}

fn same_hemisphere(w: &na::Vector3<f32>, wp: &na::Vector3<f32>) -> bool {
    w.z * wp.z > 0.0
}

pub trait BxDF {
    fn f(&self, wo: &na::Vector3<f32>, wi: &na::Vector3<f32>) -> Spectrum;
    fn sample_f(
        &self,
        wo: &na::Vector3<f32>,
        wi: &mut na::Vector3<f32>,
        u: &na::Point2<f32>,
        pdf: &mut f32,
        sampled_type: Option<BxDFType>,
    ) -> Spectrum {
        *wi = cosine_sample_hemisphere(&u);
        if wo.z < 0.0 {
            wi.z *= -1.0;
        }

        *pdf = self.pdf(&wo, &wi);
        self.f(&wo, &wi)
    }
    fn rho(&self, wo: &na::Vector3<f32>, n_samples: usize, samples: &na::Point2<f32>) -> Spectrum;
    fn rho_no_wo(
        &self,
        n_samples: usize,
        samples_1: &na::Point2<f32>,
        samples_2: &na::Point2<f32>,
    ) -> Spectrum;

    fn matches_flags(&self, t: BxDFType) -> bool {
        self.get_type().contains(t)
    }
    fn get_type(&self) -> BxDFType;
    fn pdf(&self, wo: &na::Vector3<f32>, wi: &na::Vector3<f32>) -> f32 {
        if same_hemisphere(&wo, &wi) {
            abs_cos_theta(&wi) * std::f32::consts::FRAC_1_PI
        } else {
            0.0
        }
    }
}

pub struct LambertianReflection {
    R: Spectrum,
    pub bxdf_type: BxDFType,
}

impl LambertianReflection {
    pub fn new(R: Spectrum) -> Self {
        LambertianReflection {
            R,
            bxdf_type: BxDFType::BSDF_REFLECTION | BxDFType::BSDF_DIFFUSE,
        }
    }
}

impl BxDF for LambertianReflection {
    fn f(&self, wo: &na::Vector3<f32>, wi: &na::Vector3<f32>) -> Spectrum {
        self.R * std::f32::consts::FRAC_1_PI
    }

    fn get_type(&self) -> BxDFType {
        self.bxdf_type
    }

    fn rho(&self, wo: &na::Vector3<f32>, n_samples: usize, samples: &na::Point2<f32>) -> Spectrum {
        self.R
    }

    fn rho_no_wo(
        &self,
        n_samples: usize,
        samples_1: &na::Point2<f32>,
        samples_2: &na::Point2<f32>,
    ) -> Spectrum {
        self.R
    }
}

const MAX_BXDFS: usize = 8;
pub struct BSDF {
    pub eta: f32,
    ns: na::Vector3<f32>,
    ng: na::Vector3<f32>,
    ss: na::Vector3<f32>,
    ts: na::Vector3<f32>,
    n_bxdfs: usize,
    bxdfs: [Option<Box<dyn BxDF>>; MAX_BXDFS],
}

impl BSDF {
    pub fn new(si: &SurfaceInteraction, eta: f32) -> Self {
        let ns = si.shading.n;
        let ss = si.shading.dpdu.normalize();
        BSDF {
            eta,
            ns,
            ng: si.general.n,
            ss,
            ts: ns.cross(&ss),
            n_bxdfs: 0,
            bxdfs: Default::default(),
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

    pub fn add(&mut self, b: Box<dyn BxDF>) {
        assert!(self.n_bxdfs < MAX_BXDFS);
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

    pub fn f(&self, wo_w: &na::Vector3<f32>, wi_w: &na::Vector3<f32>, flags: BxDFType) -> Spectrum {
        let wi = self.world_to_local(&wi_w);
        let wo = self.world_to_local(&wo_w);
        let reflect = wi_w.dot(&self.ng) * wo_w.dot(&self.ng) > 0.0;
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
}

pub trait Material {
    fn compute_scattering_functions(&self, si: &mut SurfaceInteraction, mode: TransportMode);
}

pub trait SyncMaterial: Material + Send + Sync {}
impl<T> SyncMaterial for T where T: Material + Send + Sync {}

pub struct MatteMaterial {}

impl Material for MatteMaterial {
    fn compute_scattering_functions(&self, si: &mut SurfaceInteraction, mode: TransportMode) {
        let mut bsdf = BSDF::new(&si, 1.0);
        let r = Spectrum::new(1.0);
        bsdf.add(Box::new(LambertianReflection::new(r)));

        si.bsdf = Some(bsdf);
    }
}
