use std::os::raw::c_void;
use nalgebra::{Matrix4, Vector3};
use opencv::core::{Mat, MatTraitConst};
use opencv::imgproc::{COLOR_RGB2BGR, cvt_color};

pub type V3d = Vector3<f64>;

pub(crate) fn get_view_matrix(eye_pos: V3d) -> Matrix4<f64> {
    let mut view: Matrix4<f64> = Matrix4::identity();
    let translate = Matrix4::new(
        1.0, 0.0, 0.0, -eye_pos.x,
        0.0, 1.0, 0.0, -eye_pos.y,
        0.0, 0.0, 1.0, -eye_pos.z,
        0.0, 0.0, 0.0, 1.0,
    );
    view = translate * view;
    view
}

pub(crate) fn get_model_matrix(rotation_angle: f64, n: V3d) -> Matrix4<f64> {
    let mut model: Matrix4<f64> = Matrix4::identity();
    let unit_n = n / f64::sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
    let mat1 = Matrix4::new(
        f64::cos(rotation_angle / 180.0 * std::f64::consts::PI), 0.0, 0.0, 0.0,
        0.0, f64::cos(rotation_angle / 180.0 * std::f64::consts::PI), 0.0, 0.0,
        0.0, 0.0, f64::cos(rotation_angle / 180.0 * std::f64::consts::PI), 0.0,
        0.0, 0.0, 0.0, 1.0,
    );
    let mat2 = (1.0 - f64::cos(rotation_angle / 180.0 * std::f64::consts::PI)) * Matrix4::new(
        unit_n.x * unit_n.x, unit_n.x * unit_n.y, unit_n.x * unit_n.z, 0.0,
        unit_n.y * unit_n.x, unit_n.y * unit_n.y, unit_n.y * unit_n.y, 0.0,
        unit_n.z * unit_n.x, unit_n.z * unit_n.y, unit_n.z * unit_n.z, 0.0,
        0.0, 0.0, 0.0, 0.0,
    );
    let mat3 = f64::sin(rotation_angle / 180.0 * std::f64::consts::PI) * Matrix4::new(
        0.0, -unit_n.z, unit_n.y, 0.0,
        unit_n.z, 0.0, -unit_n.x, 0.0,
        -unit_n.y, unit_n.x, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
    );
    model = (mat1 + mat2 + mat3) * model;
    model
}

pub(crate) fn get_projection_matrix(eye_fov: f64, aspect_ratio: f64, z_near: f64, z_far: f64) -> Matrix4<f64> {
    let mut projection: Matrix4<f64> = Matrix4::identity();
    let perspective = Matrix4::new(
        z_near, 0.0, 0.0, 0.0,
        0.0, z_near, 0.0, 0.0,
        0.0, 0.0, z_near + z_far, -z_near * z_far,
        0.0, 0.0, 1.0, 0.0,
    );
    let h = -z_near * f64::tan(eye_fov / 360.0 * std::f64::consts::PI);
    let w = h * aspect_ratio;
    let translate = Matrix4::new(
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, -(z_near + z_far) / 2.0,
        0.0, 0.0, 0.0, 1.0,
    );
    let scale = Matrix4::new(
        1.0 / w, 0.0, 0.0, 0.0,
        0.0, 1.0 / h, 0.0, 0.0,
        0.0, 0.0, 1.0 / (z_near - z_far), 0.0,
        0.0, 0.0, 0.0, 1.0,
    );
    projection = scale * translate * perspective * projection;
    projection
}

pub(crate) fn frame_buffer2cv_mat(frame_buffer: &Vec<V3d>) -> opencv::core::Mat {
    let mut image = unsafe {
        Mat::new_rows_cols_with_data(
            700, 700,
            opencv::core::CV_64FC3,
            frame_buffer.as_ptr() as *mut c_void,
            opencv::core::Mat_AUTO_STEP,
        ).unwrap()
    };
    let mut img = Mat::copy(&image).unwrap();
    image.convert_to(&mut img, opencv::core::CV_8UC3, 1.0, 1.0).expect("panic message");
    cvt_color(&img, &mut image, COLOR_RGB2BGR, 0).unwrap();
    image
}