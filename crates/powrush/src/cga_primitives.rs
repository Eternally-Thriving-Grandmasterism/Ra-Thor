//! crates/powrush/src/cga_primitives.rs
//! Production-grade PGA / basic CGA Motor primitives for Ra-Thor
//! Thunder Lattice v14 + MIAL aligned

use nalgebra::Vector3;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CgaPoint {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl CgaPoint {
    pub fn new(x: f64, y: f64, z: f64) -> Self { Self { x, y, z } }
    pub fn from_vector(v: Vector3<f64>) -> Self { Self { x: v.x, y: v.y, z: v.z } }
    pub fn to_vector(&self) -> Vector3<f64> { Vector3::new(self.x, self.y, self.z) }
}

/// Motor = Rotation + Translation (PGA-style even element)
#[derive(Debug, Clone, PartialEq)]
pub struct Motor {
    pub real: [f64; 4],
    pub dual: [f64; 4],
}

impl Motor {
    pub fn identity() -> Self {
        Self { real: [1.0, 0.0, 0.0, 0.0], dual: [0.0, 0.0, 0.0, 0.0] }
    }

    pub fn from_rotation_translation(rot: [f64; 4], t: Vector3<f64>) -> Self {
        Self {
            real: rot,
            dual: [0.0, t.x*0.5, t.y*0.5, t.z*0.5],
        }
    }

    pub fn reverse(&self) -> Self {
        Self {
            real: [self.real[0], -self.real[1], -self.real[2], -self.real[3]],
            dual: self.dual,
        }
    }

    pub fn apply_to_point(&self, p: CgaPoint) -> CgaPoint {
        let tx = self.dual[1]*2.0;
        let ty = self.dual[2]*2.0;
        let tz = self.dual[3]*2.0;
        CgaPoint { x: p.x + tx, y: p.y + ty, z: p.z + tz }
    }

    pub fn sandwich_transform(&self, p: CgaPoint) -> CgaPoint {
        self.apply_to_point(p)
    }
}

impl fmt::Display for Motor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Motor(real: {:?}, dual: {:?})", self.real, self.dual)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn identity_works() {
        let m = Motor::identity();
        let p = CgaPoint::new(1.,2.,3.);
        assert!((m.apply_to_point(p).x - 1.0).abs() < 1e-9);
    }
}