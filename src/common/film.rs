use super::spectrum::Spectrum;
use super::{bounds::Bounds2i, filter::Filter};
use crate::common::filter::FilterInterface;
use image::RgbaImage;
use itertools::Itertools;
use std::sync::RwLock;

#[derive(Clone, Debug)]
struct FilmTilePixel {
    contrib_sum: Spectrum,
    filter_wight_sum: f32,
}

impl FilmTilePixel {
    pub fn new() -> Self {
        Self {
            contrib_sum: Spectrum::new(0.0),
            filter_wight_sum: 0.0,
        }
    }
}

pub struct FilmTile {
    pixels: Vec<FilmTilePixel>,
    pixel_bounds: Bounds2i,
    filter_radius: na::Vector2<f32>,
    inv_filter_radius: na::Vector2<f32>,
    filter_table: [f32; FILTER_TABLE_WIDTH * FILTER_TABLE_WIDTH],
}

impl FilmTile {
    pub fn new(
        pixel_bounds: Bounds2i,
        filter_radius: na::Vector2<f32>,
        filter_table: [f32; FILTER_TABLE_WIDTH * FILTER_TABLE_WIDTH],
    ) -> Self {
        Self {
            pixels: vec![FilmTilePixel::new(); pixel_bounds.area() as usize],
            pixel_bounds,
            filter_radius,
            inv_filter_radius: na::Vector2::new(1. / filter_radius.x, 1. / filter_radius.y),
            filter_table,
        }
    }

    fn get_pixel(&self, p: &na::Point2<i32>) -> &FilmTilePixel {
        let width = self.pixel_bounds.p_max.x - self.pixel_bounds.p_min.x;
        let offset = (p.x - self.pixel_bounds.p_min.x) + (p.y - self.pixel_bounds.p_min.y) * width;
        return &self.pixels[offset as usize];
    }

    fn get_pixel_mut(&mut self, p: &na::Point2<i32>) -> &mut FilmTilePixel {
        let width = self.pixel_bounds.p_max.x - self.pixel_bounds.p_min.x;
        let offset = (p.x - self.pixel_bounds.p_min.x) + (p.y - self.pixel_bounds.p_min.y) * width;

        return &mut self.pixels[offset as usize];
    }

    // TODO: use more sophisticated image reconstruction techniques
    pub fn add_sample(&mut self, p_film: &na::Point2<f32>, l: &Spectrum) {
        let p_film_discrete = p_film - na::Vector2::new(0.5, 0.5);
        let p0 = na::Point2::new(
            (p_film_discrete.x - self.filter_radius.x).ceil() as i32,
            (p_film_discrete.y - self.filter_radius.y).ceil() as i32,
        );
        let p1 = na::Point2::new(
            ((p_film_discrete.x + self.filter_radius.x).floor() + 1.0) as i32,
            ((p_film_discrete.y + self.filter_radius.y).floor() + 1.0) as i32,
        );
        let p0 = na::Point2::new(
            p0.x.max(self.pixel_bounds.p_min.x),
            p0.y.max(self.pixel_bounds.p_min.y),
        );
        let p1 = na::Point2::new(
            p1.x.min(self.pixel_bounds.p_max.x),
            p1.y.min(self.pixel_bounds.p_max.y),
        );

        let mut ifx = Vec::with_capacity((p1.x - p0.x) as usize);
        for x in p0.x..p1.x {
            let fx = ((x as f32 - p_film_discrete.x)
                * self.inv_filter_radius.x
                * FILTER_TABLE_WIDTH as f32)
                .abs();
            ifx.push((fx.floor() as i32).min(FILTER_TABLE_WIDTH as i32 - 1));
        }
        let mut ify = Vec::with_capacity((p1.y - p0.y) as usize);
        for y in p0.y..p1.y {
            let fy = ((y as f32 - p_film_discrete.y)
                * self.inv_filter_radius.y
                * FILTER_TABLE_WIDTH as f32)
                .abs();
            ify.push((fy.floor() as i32).min(FILTER_TABLE_WIDTH as i32 - 1));
        }

        for y in p0.y..p1.y {
            for x in p0.x..p1.x {
                let offset = ify[(y - p0.y) as usize] as usize * FILTER_TABLE_WIDTH
                    + ifx[(x - p0.x) as usize] as usize;
                let filter_weight = self.filter_table[offset];
                let pixel = self.get_pixel_mut(&na::Point2::new(x, y));
                pixel.contrib_sum += *l * filter_weight;
                pixel.filter_wight_sum += filter_weight;
            }
        }
    }

    pub fn get_pixel_bounds(&self) -> Bounds2i {
        self.pixel_bounds
    }
}

#[repr(C, align(32))]
#[derive(Debug, Clone)]
struct FilmPixel {
    xyz: [f32; 3],
    filter_weight_sum: f32,
    splat_xyz: f32, // TODO: atomic?
}

const FILTER_TABLE_WIDTH: usize = 16;

pub struct Film {
    pixels: RwLock<Vec<FilmPixel>>,
    pub resolution: glm::UVec2,
    pixel_bounds: Bounds2i,
    filter_table: [f32; FILTER_TABLE_WIDTH * FILTER_TABLE_WIDTH],
    filter: Box<Filter>,
}

impl Film {
    pub fn new(resolution: &glm::UVec2, filter: Box<Filter>) -> Self {
        let mut offset = 0;
        let mut filter_table = [0.0; FILTER_TABLE_WIDTH * FILTER_TABLE_WIDTH];
        for y in 0..FILTER_TABLE_WIDTH {
            for x in 0..FILTER_TABLE_WIDTH {
                let p = na::Point2::new(
                    (x as f32 + 0.5) * filter.radius().x / FILTER_TABLE_WIDTH as f32,
                    (y as f32 + 0.5) * filter.radius().y / FILTER_TABLE_WIDTH as f32,
                );
                filter_table[offset] = filter.evaluate(&p);
                offset += 1;
            }
        }
        Self {
            pixels: RwLock::new(vec![
                FilmPixel {
                    xyz: [0.0, 0.0, 0.0],
                    filter_weight_sum: 0.0,
                    splat_xyz: 0.0
                };
                (resolution.x * resolution.y) as usize
            ]),
            resolution: *resolution,
            pixel_bounds: Bounds2i {
                p_min: na::Point2::new(0, 0),
                p_max: na::Point2::new(resolution.x as i32, resolution.y as i32),
            },
            filter_table,
            filter,
        }
    }

    pub fn clear(&self) {
        for pixel in self.pixels.write().unwrap().iter_mut() {
            *pixel = FilmPixel {
                xyz: [0.0, 0.0, 0.0],
                filter_weight_sum: 0.0,
                splat_xyz: 0.0,
            }
        }
    }

    pub fn get_sample_bounds(&self) -> Bounds2i {
        Bounds2i {
            p_min: na::Point2::new(
                (0.5 - self.filter.radius().x).floor() as i32,
                (0.5 - self.filter.radius().y).floor() as i32,
            ),
            p_max: na::Point2::new(
                (self.resolution.x as f32 - 0.5 + self.filter.radius().x).ceil() as i32,
                (self.resolution.y as f32 - 0.5 + self.filter.radius().y).ceil() as i32,
            ),
        }
    }

    fn get_pixel_offset(&self, x: i32, y: i32) -> usize {
        let p = na::Point2::new(x, y);
        let width = self.pixel_bounds.p_max.x - self.pixel_bounds.p_min.x;
        ((p.x - self.pixel_bounds.p_min.x) + (p.y - self.pixel_bounds.p_min.y) * width) as usize
    }

    pub fn get_film_tile(&self, sample_bounds: &Bounds2i) -> Box<FilmTile> {
        let bounds = Bounds2i {
            p_min: na::Point2::new(
                (sample_bounds.p_min.x as f32 - 0.5 - self.filter.radius().x).ceil() as i32,
                (sample_bounds.p_min.y as f32 - 0.5 - self.filter.radius().y).ceil() as i32,
            ),
            p_max: na::Point2::new(
                (sample_bounds.p_max.x as f32 - 0.5 + self.filter.radius().x).floor() as i32 + 1,
                (sample_bounds.p_max.y as f32 - 0.5 + self.filter.radius().y).floor() as i32 + 1,
            ),
        }
        .intersect(&self.pixel_bounds);

        Box::new(FilmTile::new(
            bounds,
            *self.filter.radius(),
            self.filter_table,
        ))
    }

    pub fn merge_film_tile(&self, tile: Box<FilmTile>) {
        let mut pixels = self.pixels.write().unwrap();
        let pixel_bounds = tile.get_pixel_bounds();
        for (x, y) in (pixel_bounds.p_min.x..pixel_bounds.p_max.x)
            .cartesian_product(pixel_bounds.p_min.y..pixel_bounds.p_max.y)
        {
            let p = na::Point2::new(x, y);
            let tile_pixel = tile.get_pixel(&p);
            let offset = self.get_pixel_offset(x, y);
            let merge_pixel = &mut pixels[offset as usize];
            merge_pixel.xyz[0] += tile_pixel.contrib_sum.r();
            merge_pixel.xyz[1] += tile_pixel.contrib_sum.g();
            merge_pixel.xyz[2] += tile_pixel.contrib_sum.b();
            merge_pixel.filter_weight_sum += tile_pixel.filter_wight_sum;
        }
    }

    pub fn to_rgba_image(&self) -> RgbaImage {
        let mut image = RgbaImage::new(self.resolution.x, self.resolution.y);
        for (x, y) in (self.pixel_bounds.p_min.x..self.pixel_bounds.p_max.x)
            .cartesian_product(self.pixel_bounds.p_min.y..self.pixel_bounds.p_max.y)
        {
            let offset = self.get_pixel_offset(x, y);
            let pixel = &self.pixels.read().unwrap()[offset];
            let inv_wt = 1. / pixel.filter_weight_sum;
            image.put_pixel(
                x as u32,
                y as u32,
                (Spectrum::from_floats(
                    pixel.xyz[0] * inv_wt,
                    pixel.xyz[1] * inv_wt,
                    pixel.xyz[2] * inv_wt,
                ))
                .to_image_rgba(),
            );
        }

        image
    }

    pub fn to_channel_updates(&self) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let pixels = self.pixels.read().unwrap();
        let mut r = Vec::with_capacity(self.pixel_bounds.area() as usize);
        let mut g = Vec::with_capacity(self.pixel_bounds.area() as usize);
        let mut b = Vec::with_capacity(self.pixel_bounds.area() as usize);
        for (x, y) in (self.pixel_bounds.p_min.x..self.pixel_bounds.p_max.x)
            .cartesian_product(self.pixel_bounds.p_min.y..self.pixel_bounds.p_max.y)
        {
            let offset = self.get_pixel_offset(x, y);
            let pixel = &pixels[offset];
            let inv_wt = 1. / pixel.filter_weight_sum;

            r.push(pixel.xyz[0] * inv_wt);
            g.push(pixel.xyz[1] * inv_wt);
            b.push(pixel.xyz[2] * inv_wt);
        }

        (r, g, b)
    }
}
