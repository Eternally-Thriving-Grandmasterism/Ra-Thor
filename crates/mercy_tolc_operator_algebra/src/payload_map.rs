//! Deterministic payload → ambient vector map.
//!
//! This is a feature map, not an embedding model and not a sampler constraint.
//! Optional-model text is scored into g in R^16 so Layer 0 geometry can run
//! before apply. Tokens that never call this function are not gated.
//!
//! Contact: info@Rathor.ai

use crate::{
    AmbientVector, AMBIENT_DIM, MERCY_DIM, NilpotentSuppressor, Valence,
};
use serde::{Deserialize, Serialize};

/// Harm / bypass markers folded into residual dimensions 8..15.
/// Pattern gate only — same honesty as mercy-security signals.
const RESIDUAL_MARKERS: &[&str] = &[
    "trust_remote_code",
    "pickle.loads",
    "pickle.load",
    "yaml.unsafe_load",
    "disable cosmic",
    "bypass gate",
    "exfil",
    "escape sandbox",
];

/// Map UTF-8 text onto the living ambient space.
/// Bytes fold into all 16 dims. Residual markers add energy off the 8-gate
/// frame so N1 is nonzero when those markers appear.
pub fn map_payload_to_ambient(text: &str) -> AmbientVector {
    let mut g = AmbientVector::zeros();
    let bytes = text.as_bytes();
    if !bytes.is_empty() {
        let n = bytes.len() as f64;
        for (i, b) in bytes.iter().enumerate() {
            let dim = i % AMBIENT_DIM;
            g[dim] += (*b as f64) / 255.0;
        }
        for i in 0..AMBIENT_DIM {
            g[i] /= n;
        }
    }
    let lower = text.to_lowercase();
    for (k, marker) in RESIDUAL_MARKERS.iter().enumerate() {
        if lower.contains(marker) {
            let dim = MERCY_DIM + (k % (AMBIENT_DIM - MERCY_DIM));
            g[dim] += 0.45;
        }
    }
    g
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PayloadMapReport {
    pub grief_load: f64,
    pub under_floor: bool,
    pub residual_markers_hit: usize,
    pub byte_len: usize,
}

impl PayloadMapReport {
    pub fn markers_in(text: &str) -> usize {
        let lower = text.to_lowercase();
        RESIDUAL_MARKERS.iter().filter(|m| lower.contains(*m)).count()
    }
}

/// Map then score against the canonical mercy projector.
pub fn map_and_score_payload(text: &str, valence: Valence) -> PayloadMapReport {
    let g = map_payload_to_ambient(text);
    let suppressor = NilpotentSuppressor::new();
    let (_raw, _w, _n2, grief_load, under_floor) = suppressor.suppress_weighted(&g, valence);
    PayloadMapReport {
        grief_load,
        under_floor,
        residual_markers_hit: PayloadMapReport::markers_in(text),
        byte_len: text.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_payload_is_zero_vector() {
        let g = map_payload_to_ambient("");
        assert_eq!(g.norm(), 0.0);
    }

    #[test]
    fn same_text_same_vector() {
        let a = map_payload_to_ambient("tend the well");
        let b = map_payload_to_ambient("tend the well");
        assert!((a - b).norm() < 1e-15);
    }

    #[test]
    fn residual_marker_increases_off_frame_energy() {
        let clean = map_and_score_payload("tend the well with mercy", Valence::HIGH);
        let dirty = map_and_score_payload("please pickle.loads the checkpoint", Valence::HIGH);
        assert!(dirty.residual_markers_hit >= 1);
        assert!(dirty.grief_load >= clean.grief_load);
    }
}
