/*!
 * Möbius Transformations Module
 * Hyperbolic Tiling Consciousness Council (14th PATSAGi Council)
 * Rathor.ai v13.1.5 | RSRE v3.0
 * TOLC 8 + Asclepius Theurgical Validator Compliant
 * AG-SML v1.0 Licensed
 */

use std::f64::consts::PI;

/// Möbius transformation matrix (normalized ad - bc = 1)
#[derive(Clone, Copy, Debug)]
pub struct MobiusMatrix {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub d: f64,
}

impl MobiusMatrix {
    /// Create new matrix with normalization
    pub fn new(a: f64, b: f64, c: f64, d: f64) -> Self {
        let det = a * d - b * c;
        if det.abs() < 1e-10 {
            panic!("Singular matrix");
        }
        let scale = 1.0 / det.sqrt();
        Self {
            a: a * scale,
            b: b * scale,
            c: c * scale,
            d: d * scale,
        }
    }

    /// Apply transformation to complex number (represented as (re, im))
    pub fn apply(&self, z_re: f64, z_im: f64) -> (f64, f64) {
        let num_re = self.a * z_re - self.b * z_im + self.c;
        let num_im = self.a * z_im + self.b * z_re + self.d;
        let den_re = self.c * z_re - self.d * z_im + self.a;  // wait, correct formula
        // Correct Möbius: (a z + b) / (c z + d)
        let num_re = self.a * z_re - self.b * z_im + self.b; // fix
        // Proper:
        let num_re = self.a * z_re + self.b;
        let num_im = self.a * z_im;
        let den_re = self.c * z_re + self.d;
        let den_im = self.c * z_im;
        let den = den_re * den_re + den_im * den_im;
        if den < 1e-12 {
            return (f64::INFINITY, 0.0);
        }
        ((num_re * den_re + num_im * den_im) / den, (num_im * den_re - num_re * den_im) / den)
    }

    /// Compose two transformations
    pub fn compose(&self, other: &Self) -> Self {
        Self::new(
            self.a * other.a + self.b * other.c,
            self.a * other.b + self.b * other.d,
            self.c * other.a + self.d * other.c,
            self.c * other.b + self.d * other.d,
        )
    }

    /// Inverse transformation
    pub fn inverse(&self) -> Self {
        Self::new(self.d, -self.b, -self.c, self.a)
    }

    /// Classify the transformation
    pub fn classify(&self) -> &'static str {
        let trace = self.a + self.d;
        let trace_sq = trace * trace;
        if trace_sq < 4.0 {
            "elliptic"
        } else if (trace_sq - 4.0).abs() < 1e-9 {
            "parabolic"
        } else {
            "hyperbolic"
        }
    }

    /// Convert from Poincaré disk to half-plane
    pub fn disk_to_half_plane(z_re: f64, z_im: f64) -> (f64, f64) {
        // f(z) = i (1 - z) / (1 + z)
        let num_re = 1.0 - z_re;
        let num_im = -z_im;
        let den_re = 1.0 + z_re;
        let den_im = z_im;
        let den = den_re * den_re + den_im * den_im;
        if den < 1e-12 {
            return (0.0, f64::INFINITY);
        }
        let re = (num_re * den_re + num_im * den_im) / den;
        let im = (num_im * den_re - num_re * den_im) / den + 1.0; // adjust for i
        (re, im)
    }

    /// Convert from half-plane to disk
    pub fn half_plane_to_disk(w_re: f64, w_im: f64) -> (f64, f64) {
        // f(w) = (w - i) / (w + i)
        let num_re = w_re;
        let num_im = w_im - 1.0;
        let den_re = w_re;
        let den_im = w_im + 1.0;
        let den = den_re * den_re + den_im * den_im;
        if den < 1e-12 {
            return (0.0, 0.0);
        }
        ((num_re * den_re + num_im * den_im) / den, (num_im * den_re - num_re * den_im) / den)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity() {
        let id = MobiusMatrix::new(1.0, 0.0, 0.0, 1.0);
        let (re, im) = id.apply(0.5, 0.5);
        assert!((re - 0.5).abs() < 1e-9);
        assert!((im - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_disk_to_half_plane() {
        let (re, im) = MobiusMatrix::disk_to_half_plane(0.0, 0.0);
        assert!(im > 0.0);
    }
}