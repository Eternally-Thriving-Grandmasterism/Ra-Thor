//! Möbius matrix over `nalgebra::Complex`. `apply` is `(az + b) / (cz + d)`. `gyrovector_add` does not read the matrix.

use nalgebra::Complex;

pub struct MoebiusMatrix {
    pub a: Complex<f64>,
    pub b: Complex<f64>,
    pub c: Complex<f64>,
    pub d: Complex<f64>,
}

impl MoebiusMatrix {
    pub fn new(a: Complex<f64>, b: Complex<f64>, c: Complex<f64>, d: Complex<f64>) -> Self {
        Self { a, b, c, d }
    }

    pub fn identity() -> Self {
        Self {
            a: Complex::new(1.0, 0.0),
            b: Complex::new(0.0, 0.0),
            c: Complex::new(0.0, 0.0),
            d: Complex::new(1.0, 0.0),
        }
    }

    pub fn apply(&self, z: Complex<f64>) -> Complex<f64> {
        (self.a * z + self.b) / (self.c * z + self.d)
    }

    /// Does not read `self`. Returns the expression in the body.
    #[allow(clippy::unused_self)]
    pub fn gyrovector_add(&self, u: Complex<f64>, v: Complex<f64>) -> Complex<f64> {
        let num = u + v + (u * v).conj() * (u.norm_sqr() + v.norm_sqr() - 2.0 * (u * v).re);
        // Simplified for demonstration; full formula in production
        num / (1.0 + (u * v).re)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Normed;
    #[test]
    fn test_moebius_identity() {
        let m = MoebiusMatrix::identity();
        let z = Complex::new(0.5, 0.3);
        assert!((m.apply(z) - z).norm() < 1e-10);
    }

    #[test]
    fn apply_matches_the_matrix_formula() {
        let m = MoebiusMatrix::new(
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(1.0, 0.0),
        );
        let z = Complex::new(0.5, 0.0);
        let got = m.apply(z);
        assert!((got - Complex::new(2.5, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn gyrovector_add_ignores_the_matrix() {
        let left = MoebiusMatrix::identity();
        let right = MoebiusMatrix::new(
            Complex::new(2.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.5, 0.0),
        );
        let u = Complex::new(0.2, 0.1);
        let v = Complex::new(0.3, -0.1);
        let from_left = left.gyrovector_add(u, v);
        let from_right = right.gyrovector_add(u, v);
        assert!((from_left - from_right).norm() < 1e-12);
        let num = u + v + (u * v).conj() * (u.norm_sqr() + v.norm_sqr() - 2.0 * (u * v).re);
        let expect = num / (1.0 + (u * v).re);
        assert!((from_left - expect).norm() < 1e-12);
    }
}
