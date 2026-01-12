use super::lab::{lab_distance, rgb_to_lab};
use super::Rgb;

pub(super) const ANSI_256_MAX: u8 = 255;

pub(super) fn nearest_index(rgb: Rgb, palette_size: u16) -> u8 {
    let mut best = 0u8;
    let mut best_distance = f32::MAX;
    let max = palette_size.saturating_sub(1).min(u16::from(ANSI_256_MAX));
    for index in 0..=max {
        let candidate = ansi_color(index as u8);
        let distance = lab_distance(rgb_to_lab(rgb), rgb_to_lab(candidate));
        if distance < best_distance {
            best_distance = distance;
            best = index as u8;
        }
    }
    best
}

fn ansi_color(index: u8) -> Rgb {
    if index < ANSI16.len() as u8 {
        return ANSI16[index as usize];
    }
    ansi256_color(index)
}

fn ansi256_color(index: u8) -> Rgb {
    if index < ANSI16.len() as u8 {
        return ANSI16[index as usize];
    }
    if index <= ANSI_256_COLOR_MAX {
        let offset = index - ANSI_256_FIRST;
        let r = offset / ANSI_256_STRIDE;
        let g = (offset % ANSI_256_STRIDE) / ANSI_256_STEP;
        let b = offset % ANSI_256_STEP;
        return Rgb::new(
            ANSI_256_LEVELS[r as usize],
            ANSI_256_LEVELS[g as usize],
            ANSI_256_LEVELS[b as usize],
        );
    }
    let gray = ANSI_256_GRAY_FIRST + (index - ANSI_256_GRAY_INDEX) * ANSI_256_GRAY_STEP;
    Rgb::new(gray, gray, gray)
}

const ANSI_256_FIRST: u8 = 16;
const ANSI_256_COLOR_MAX: u8 = 231;
const ANSI_256_GRAY_INDEX: u8 = 232;
const ANSI_256_GRAY_FIRST: u8 = 8;
const ANSI_256_GRAY_STEP: u8 = 10;
const ANSI_256_STRIDE: u8 = 36;
const ANSI_256_STEP: u8 = 6;
const ANSI_256_LEVELS: [u8; 6] = [0, 95, 135, 175, 215, 255];

const ANSI16: [Rgb; 16] = [
    Rgb::new(0, 0, 0),
    Rgb::new(205, 0, 0),
    Rgb::new(0, 205, 0),
    Rgb::new(205, 205, 0),
    Rgb::new(0, 0, 238),
    Rgb::new(205, 0, 205),
    Rgb::new(0, 205, 205),
    Rgb::new(229, 229, 229),
    Rgb::new(127, 127, 127),
    Rgb::new(255, 0, 0),
    Rgb::new(0, 255, 0),
    Rgb::new(255, 255, 0),
    Rgb::new(92, 92, 255),
    Rgb::new(255, 0, 255),
    Rgb::new(0, 255, 255),
    Rgb::new(255, 255, 255),
];
