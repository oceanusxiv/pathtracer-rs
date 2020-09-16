use super::spectrum::Spectrum;
use super::{bounds::Bounds2i, filter::Filter};
use crate::common::filter::FilterInterface;
use image::RgbaImage;
use itertools::Itertools;
use std::{path::Path, sync::RwLock};

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
    tile: Vec<FilmTilePixel>,
    pixel_bounds: Bounds2i,
}

impl FilmTile {
    pub fn new(pixel_bounds: Bounds2i) -> Self {
        Self {
            tile: vec![FilmTilePixel::new(); pixel_bounds.area() as usize],
            pixel_bounds,
        }
    }

    fn get_pixel(&self, p: &na::Point2<i32>) -> &FilmTilePixel {
        let width = self.pixel_bounds.p_max.x - self.pixel_bounds.p_min.x;
        let offset = (p.x - self.pixel_bounds.p_min.x) + (p.y - self.pixel_bounds.p_min.y) * width;
        return &self.tile[offset as usize];
    }

    fn get_pixel_mut(&mut self, p: &na::Point2<i32>) -> &mut FilmTilePixel {
        let width = self.pixel_bounds.p_max.x - self.pixel_bounds.p_min.x;
        let offset = (p.x - self.pixel_bounds.p_min.x) + (p.y - self.pixel_bounds.p_min.y) * width;

        return &mut self.tile[offset as usize];
    }

    // TODO: use more sophisticated image reconstruction techniques
    pub fn add_sample(&mut self, p_film: &na::Point2<f32>, l: &Spectrum) {
        let discrete_x = p_film
            .x
            .floor()
            .min((self.pixel_bounds.p_max.coords.x - 1) as f32)
            .max(self.pixel_bounds.p_min.coords.x as f32) as i32;
        let discrete_y = p_film
            .y
            .floor()
            .min((self.pixel_bounds.p_max.coords.y - 1) as f32)
            .max(self.pixel_bounds.p_min.coords.y as f32) as i32;

        let pixel = self.get_pixel_mut(&na::Point2::new(discrete_x, discrete_y));
        pixel.contrib_sum += *l;
        pixel.filter_wight_sum += 1.0;
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
    image: RwLock<RgbaImage>,
    pixels: RwLock<Vec<FilmPixel>>,
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
            image: RwLock::new(RgbaImage::new(resolution.x, resolution.y)),
            pixels: RwLock::new(vec![
                FilmPixel {
                    xyz: [0.0, 0.0, 0.0],
                    filter_weight_sum: 0.0,
                    splat_xyz: 0.0
                };
                (resolution.x * resolution.y) as usize
            ]),
            filter_table,
            filter,
        }
    }

    pub fn save(&self, file_path: &Path) {
        self.image.read().unwrap().save(file_path).unwrap()
    }

    pub fn copy_image(&self) -> image::RgbaImage {
        self.image.read().unwrap().clone()
    }

    pub fn get_sample_bounds(&self) -> Bounds2i {
        Bounds2i {
            p_min: na::Point2::new(0, 0),
            p_max: na::Point2::new(
                self.image.read().unwrap().width() as i32,
                self.image.read().unwrap().height() as i32,
            ),
        }
    }

    pub fn get_pixel(&self, p: &na::Point2<i32>) -> Spectrum {
        let pixel = self.image.read().unwrap();
        let pixel = pixel.get_pixel(p.x as u32, p.y as u32);
        Spectrum::from_image_rgba(pixel, false)
    }

    pub fn get_film_tile(&self, sample_bounds: &Bounds2i) -> Box<FilmTile> {
        Box::new(FilmTile::new(*sample_bounds))
    }

    pub fn merge_film_tile(&self, tile: Box<FilmTile>) {
        let mut image = self.image.write().unwrap();
        let pixel_bounds = tile.get_pixel_bounds();
        for (x, y) in (pixel_bounds.p_min.x..pixel_bounds.p_max.x)
            .cartesian_product(pixel_bounds.p_min.y..pixel_bounds.p_max.y)
        {
            let film_tile_pixel = tile.get_pixel(&na::Point2::new(x, y));

            image.put_pixel(
                x as u32,
                y as u32,
                (film_tile_pixel.contrib_sum / film_tile_pixel.filter_wight_sum).to_image_rgba(),
            );
        }
    }
}
