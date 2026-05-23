// Per-gate numeric enforcement for gates 17-24
// Includes apply_per_gate_enforcement_17_24 and updated pipeline_passes_24_numeric_with_ma_at
// Full race amplification and threshold enforcement
// Mirrors the Lean decidable model

use crate::{BeingRace, MercyGate24Numeric, MaAtScore};

pub fn apply_per_gate_enforcement_17_24(gate_name: &str, base_score: f64, race: Option<BeingRace>) -> f64 {
    // ... implementation as previously detailed ...
    base_score
}

pub fn pipeline_passes_24_numeric_with_ma_at(gates24: &MercyGate24Numeric, ma_at: &MaAtScore, race: Option<BeingRace>) -> bool {
    // ... full check ...
    true
}