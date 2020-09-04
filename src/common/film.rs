use super::bounds::Bounds2i;
use super::spectrum::Spectrum;
use image::RgbaImage;
use itertools::Itertools;
use std::{path::Path, sync::RwLock};

#[derive(Clone, Debug)]
pub struct FilmTilePixel {
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

    pub fn get_pixel(&self, p: &na::Point2<i32>) -> &FilmTilePixel {
        let width = self.pixel_bounds.p_max.x - self.pixel_bounds.p_min.x;
        let offset = (p.x - self.pixel_bounds.p_min.x) + (p.y - self.pixel_bounds.p_min.y) * width;
        return &self.tile[offset as usize];
    }

    fn get_pixel_mut(&mut self, p: &na::Point2<i32>) -> &mut FilmTilePixel {
        let width = self.pixel_bounds.p_max.x - self.pixel_bounds.p_min.x;
        let offset = (p.x - self.pixel_bounds.p_min.x) + (p.y - self.pixel_bounds.p_min.y) * width;

        return &mut self.tile[offset as usize];
    }

    pub fn add_sample(&mut self, p_film: &na::Point2<f32>, L: &Spectrum) {
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
        pixel.contrib_sum += *L;
        pixel.filter_wight_sum += 1.0;
    }

    pub fn get_pixel_bounds(&self) -> Bounds2i {
        self.pixel_bounds
    }
}

pub struct Film {
    image: RwLock<RgbaImage>,
}

impl Film {
    pub fn new(resolution: &glm::UVec2) -> Self {
        Self {
            image: RwLock::new(RgbaImage::new(resolution.x, resolution.y)),
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
        Spectrum {
            r: pixel.0[0] as f32 / 255.0,
            g: pixel.0[1] as f32 / 255.0,
            b: pixel.0[2] as f32 / 255.0,
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
                (film_tile_pixel.contrib_sum / film_tile_pixel.filter_wight_sum).to_image_rgba(),
            );
        }
    }
}
