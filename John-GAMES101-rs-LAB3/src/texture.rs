use nalgebra::{Vector3};
use opencv::core::{MatTraitConst, VecN};
use opencv::imgcodecs::{imread, IMREAD_COLOR};

pub struct Texture {
    pub img_data: opencv::core::Mat,
    pub width: usize,
    pub height: usize,
}

impl Texture {
    pub fn new(name: &str) -> Self {
        let img_data = imread(name, IMREAD_COLOR).expect("Image reading error!");
        let width = img_data.cols() as usize;
        let height = img_data.rows() as usize;
        Texture {
            img_data,
            width,
            height,
        }
    }

    pub fn getColor(&self, mut u: f64, mut v: f64) -> Vector3<f64> {
        let u_img = u * self.width as f64;
        let v_img = (1.0 - v) * self.height as f64;
        let color: &VecN<u8, 3> = self.img_data.at_2d(v_img as i32, u_img as i32).unwrap();

        Vector3::new(color[0] as f64, color[1] as f64, color[2] as f64)
    }

    pub fn getColorBilinear(&self, mut u: f64, mut v: f64) -> Vector3<f64> {
        let u_img = u * self.width as f64;
        let v_img = (1.0 - v) * self.height as f64;
        let u_min = f64::floor(u_img) as i32;
        let u_max = i32::min(f64::ceil(u_img) as i32, self.width as i32);
        let v_min = f64::floor(v_img) as i32;
        let v_max = i32::min(f64::ceil(v_img) as i32, self.height as i32);
        let color1: VecN<f64, 3> = *self.img_data.at_2d(v_max, u_min).unwrap();
        let color2: VecN<f64, 3> = *self.img_data.at_2d(v_max, u_max).unwrap();
        let color3: VecN<f64, 3> = *self.img_data.at_2d(v_min, u_min).unwrap();
        let color4: VecN<f64, 3> = *self.img_data.at_2d(v_min, u_max).unwrap();
        let ratio_u = (u_img - u_min as f64) / (u_max - u_min) as f64;
        let ratio_v = (v_img - v_min as f64) / (v_max - v_min) as f64;
        let up =  color3 * (1.0 - ratio_u) + color4 * ratio_u;
        let down = color1 * (1.0 - ratio_u) + color2 * ratio_u;
        let color = up * (1.0 - ratio_v) + down * ratio_v;
        Vector3::new(color[0] as f64, color[1] as f64, color[2] as f64)
    }
}