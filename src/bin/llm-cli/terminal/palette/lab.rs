use super::Rgb;

#[derive(Clone, Copy)]
pub(super) struct Lab {
    l: f32,
    a: f32,
    b: f32,
}

pub(super) fn blend_lab(a: Rgb, b: Rgb, t: f32) -> Rgb {
    let a_lab = rgb_to_lab(a);
    let b_lab = rgb_to_lab(b);
    let lab = Lab {
        l: lerp(a_lab.l, b_lab.l, t),
        a: lerp(a_lab.a, b_lab.a, t),
        b: lerp(a_lab.b, b_lab.b, t),
    };
    lab_to_rgb(lab)
}

pub(super) fn rgb_to_lab(rgb: Rgb) -> Lab {
    let (x, y, z) = rgb_to_xyz(rgb);
    xyz_to_lab(x, y, z)
}

pub(super) fn lab_distance(a: Lab, b: Lab) -> f32 {
    let dl = a.l - b.l;
    let da = a.a - b.a;
    let db = a.b - b.b;
    (dl * dl + da * da + db * db).sqrt()
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn lab_to_rgb(lab: Lab) -> Rgb {
    let (x, y, z) = lab_to_xyz(lab);
    xyz_to_rgb(x, y, z)
}

fn rgb_to_xyz(rgb: Rgb) -> (f32, f32, f32) {
    let r = srgb_to_linear(rgb.r);
    let g = srgb_to_linear(rgb.g);
    let b = srgb_to_linear(rgb.b);
    let x = (r * RGB_TO_XYZ[0]) + (g * RGB_TO_XYZ[1]) + (b * RGB_TO_XYZ[2]);
    let y = (r * RGB_TO_XYZ[3]) + (g * RGB_TO_XYZ[4]) + (b * RGB_TO_XYZ[5]);
    let z = (r * RGB_TO_XYZ[6]) + (g * RGB_TO_XYZ[7]) + (b * RGB_TO_XYZ[8]);
    (x, y, z)
}

fn xyz_to_rgb(x: f32, y: f32, z: f32) -> Rgb {
    let r = (x * XYZ_TO_RGB[0]) + (y * XYZ_TO_RGB[1]) + (z * XYZ_TO_RGB[2]);
    let g = (x * XYZ_TO_RGB[3]) + (y * XYZ_TO_RGB[4]) + (z * XYZ_TO_RGB[5]);
    let b = (x * XYZ_TO_RGB[6]) + (y * XYZ_TO_RGB[7]) + (z * XYZ_TO_RGB[8]);
    let r = linear_to_srgb(r);
    let g = linear_to_srgb(g);
    let b = linear_to_srgb(b);
    Rgb::new(r, g, b)
}

fn srgb_to_linear(value: u8) -> f32 {
    let normalized = value as f32 / SRGB_MAX;
    if normalized <= SRGB_EPS {
        normalized / SRGB_DIV
    } else {
        ((normalized + SRGB_OFF) / SRGB_SCALE).powf(SRGB_GAMMA)
    }
}

fn linear_to_srgb(value: f32) -> u8 {
    let clamped = value.clamp(0.0, 1.0);
    let encoded = if clamped <= LIN_EPS {
        clamped * SRGB_DIV
    } else {
        SRGB_SCALE * clamped.powf(1.0 / SRGB_GAMMA) - SRGB_OFF
    };
    (encoded * SRGB_MAX).round().clamp(0.0, SRGB_MAX) as u8
}

fn xyz_to_lab(x: f32, y: f32, z: f32) -> Lab {
    let x = lab_f(x / D65_WHITE[0]);
    let y = lab_f(y / D65_WHITE[1]);
    let z = lab_f(z / D65_WHITE[2]);
    Lab {
        l: (LAB_SCALE * y) - LAB_OFF,
        a: LAB_A_SCALE * (x - y),
        b: LAB_B_SCALE * (y - z),
    }
}

fn lab_to_xyz(lab: Lab) -> (f32, f32, f32) {
    let y = (lab.l + LAB_OFF) / LAB_SCALE;
    let x = lab.a / LAB_A_SCALE + y;
    let z = y - lab.b / LAB_B_SCALE;
    (
        D65_WHITE[0] * lab_inv_f(x),
        D65_WHITE[1] * lab_inv_f(y),
        D65_WHITE[2] * lab_inv_f(z),
    )
}

fn lab_f(value: f32) -> f32 {
    if value > LAB_EPS {
        value.powf(1.0 / LAB_KAPPA)
    } else {
        (LAB_KAPPA * value + LAB_OFF) / LAB_SCALE
    }
}

fn lab_inv_f(value: f32) -> f32 {
    let cube = value.powi(3);
    if cube > LAB_EPS {
        cube
    } else {
        (LAB_SCALE * value - LAB_OFF) / LAB_KAPPA
    }
}

const SRGB_MAX: f32 = 255.0;
const SRGB_EPS: f32 = 0.04045;
const SRGB_DIV: f32 = 12.92;
const SRGB_OFF: f32 = 0.055;
const SRGB_SCALE: f32 = 1.055;
const SRGB_GAMMA: f32 = 2.4;
const LIN_EPS: f32 = 0.0031308;

const D65_WHITE: [f32; 3] = [95.047, 100.0, 108.883];
const LAB_EPS: f32 = 0.008856;
const LAB_KAPPA: f32 = 7.787;
const LAB_SCALE: f32 = 116.0;
const LAB_OFF: f32 = 16.0;
const LAB_A_SCALE: f32 = 500.0;
const LAB_B_SCALE: f32 = 200.0;

const RGB_TO_XYZ: [f32; 9] = [41.24, 35.76, 18.05, 21.26, 71.52, 7.22, 1.93, 11.92, 95.05];
const XYZ_TO_RGB: [f32; 9] = [
    3.2406, -1.5372, -0.4986, -0.9689, 1.8758, 0.0415, 0.0557, -0.2040, 1.0570,
];
