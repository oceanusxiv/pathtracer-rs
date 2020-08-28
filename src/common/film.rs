use super::bounds::Bounds2i;
use super::spectrum::Spectrum;
use image::RgbImage;
use itertools::Itertools;
use std::{
    path::Path,
    sync::{Arc, RwLock},
};

#[derive(Clone, Debug)]
pub struct FilmTilePixel {
    contrib_sum: Spectrum,
}

impl FilmTilePixel {
    pub fn new() -> Self {
        FilmTilePixel {
            contrib_sum: Spectrum::new(0.0),
        }
    }
}

pub struct FilmTile {
    tile: Vec<FilmTilePixel>,
    pixel_bounds: Bounds2i,
}

impl FilmTile {
    pub fn new(pixel_bounds: Bounds2i) -> Self {
        FilmTile {
            tile: vec![FilmTilePixel::new(); pixel_bounds.area() as usize],
            pixel_bounds,
        }
    }

    pub fn get_pixel(&self, p: &na::Point2<i32>) -> &FilmTilePixel {
        let width = self.pixel_bounds.p_max.x - self.pixel_bounds.p_min.x;
        let offset = (p.x - self.pixel_bounds.p_min.x) + (p.y - self.pixel_bounds.p_min.y) * width;
        return &self.tile[offset as usize];
    }

    pub fn get_pixel_mut(&mut self, p: &na::Point2<i32>) -> &mut FilmTilePixel {
        let width = self.pixel_bounds.p_max.x - self.pixel_bounds.p_min.x;
        let offset = (p.x - self.pixel_bounds.p_min.x) + (p.y - self.pixel_bounds.p_min.y) * width;
        return &mut self.tile[offset as usize];
    }

    pub fn add_sample(&mut self, p_film: &na::Point2<f32>, L: &Spectrum) {
        let p_film_discrete = p_film - na::Vector2::new(0.5, 0.5);
        let pixel = self.get_pixel_mut(&na::Point2::new(
            p_film_discrete.x.floor() as i32,
            p_film_discrete.y.floor() as i32,
        ));
        pixel.contrib_sum += *L;
    }

    pub fn get_pixel_bounds(&self) -> Bounds2i {
        self.pixel_bounds
    }
}

pub struct Film {
    image: RwLock<RgbImage>,
}

impl Film {
    pub fn new(resolution: &glm::UVec2) -> Self {
        Film {
            image: RwLock::new(RgbImage::new(resolution.x, resolution.y)),
        }
    }

    pub fn save(&self, file_path: &Path) {
        self.image.read().unwrap().save(file_path).unwrap()
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

    pub fn get_film_tile(&self, sample_bounds: &Bounds2i) -> Box<FilmTile> {
        Box::new(FilmTile::new(*sample_bounds))
    }

    pub fn merge_film_tile(&mut self, tile: Box<FilmTile>) {
        let mut image = self.image.write().unwrap();
        let pixel_bounds = tile.get_pixel_bounds();
        for (x, y) in (pixel_bounds.p_min.x..pixel_bounds.p_max.x)
            .cartesian_product(pixel_bounds.p_min.y..pixel_bounds.p_max.y)
        {
            let film_tile_pixel = tile.get_pixel(&na::Point2::new(x, y));

            image.put_pixel(
                x as u32,
                y as u32,
                film_tile_pixel.contrib_sum.to_image_rgb(),
            );
        }
    }
}
