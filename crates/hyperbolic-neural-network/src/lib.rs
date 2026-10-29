//! `project_to_hyperbolic` writes `tanh` of an input prefix.
//! `forward` returns Err when `valence` is below the threshold constant 0.9999999 (a source literal, not a measured value).

pub struct HyperbolicNN {
    pub embedding_dim: usize,
    pub curvature: f64, // -1.0 for hyperbolic
}

impl HyperbolicNN {
    pub fn new(embedding_dim: usize) -> Self {
        Self {
            embedding_dim,
            curvature: -1.0,
        }
    }

    /// Writes `tanh` of each input entry that fits in `embedding_dim`. The rest stay 0.
    pub fn project_to_hyperbolic(&self, input: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.embedding_dim];
        for (i, &x) in input.iter().enumerate() {
            if i < self.embedding_dim {
                out[i] = x.tanh(); // Approximate hyperbolic activation
            }
        }
        out
    }

    /// Returns Err when `valence` is below the threshold constant 0.9999999 (a source literal, not a measured value).
    pub fn forward(&self, input: &[f64], valence: f64) -> Result<Vec<f64>, String> {
        if valence < 0.9999999 {
            return Err("TOLC 8 violation: valence below threshold".to_string());
        }
        Ok(self.project_to_hyperbolic(input))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_hyperbolic_projection() {
        let nn = HyperbolicNN::new(128);
        let input = vec![0.5; 10];
        let out = nn.project_to_hyperbolic(&input);
        assert_eq!(out.len(), 128);
    }

    #[test]
    fn test_tolc8_forward() {
        let nn = HyperbolicNN::new(64);
        let input = vec![0.8; 5];
        let result = nn.forward(&input, 0.99999999);
        assert!(result.is_ok());
    }

    #[test]
    fn project_writes_tanh_of_the_prefix() {
        let nn = HyperbolicNN::new(3);
        let out = nn.project_to_hyperbolic(&[0.0, 1.0]);
        assert_eq!(out.len(), 3);
        assert!((out[0] - 0.0_f64.tanh()).abs() < 1e-12);
        assert!((out[1] - 1.0_f64.tanh()).abs() < 1e-12);
        assert_eq!(out[2], 0.0);
    }

    #[test]
    fn forward_rejects_valence_below_the_source_literal() {
        let nn = HyperbolicNN::new(2);
        let err = nn.forward(&[0.25], 0.0).unwrap_err();
        assert_eq!(err, "TOLC 8 violation: valence below threshold");
    }

    #[test]
    fn forward_at_the_source_literal_returns_the_projection() {
        let nn = HyperbolicNN::new(1);
        let out = nn
            .forward(&[0.0], 0.9999999)
            .expect("the literal is not below itself");
        assert_eq!(out, vec![0.0]);
    }
}
