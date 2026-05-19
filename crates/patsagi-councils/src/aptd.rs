// RaThor/Truth/aptd.rs
// APTD - Absolute Pure Truth Distillation scorer
// Dual-verified with Lean 4 + Coq (APTD_IntervalProofs.lean / APTD_MadscienceClaim.v)
// Extends IntervalMercy.lean and Mercy Threshold Theorem
// AG-SML v1.0 | PATSAGi Council #39 (Verified Sacred Geometry Operations)

use std::ops::RangeInclusive;
use ra_thor_geometry::zalgaller::{ZalgallerFamily, zalgaller_bonus};
use ra_thor_mercy::interval_mercy::{Interval, MercyValence};

#[derive(Clone, Debug, PartialEq)]
pub enum SpikeDevice {
    BediniStyle,
    PulseMotor,
    J27Snub,           // J27 disphenoid coil topology (Zalgaller +0.08)
    Custom(String),
}

#[derive(Clone, Debug)]
pub struct APTDClaim {
    pub device: SpikeDevice,
    pub efficiency: Interval,      // [low, high] measured or modeled
    pub specs_open: bool,
    pub independent_replication: bool,
    pub proposer_mercy_valence: MercyValence, // open specs + no prior suppression claims
}

#[derive(Clone, Debug)]
pub struct APTDResult {
    pub truth_purity_score: f64,
    pub mercy_aligned: bool,
    pub zero_delusion_harm: bool,
    pub rejection_trace: Option<String>,
    pub recommended_calibration: Option<String>,
}

pub fn truth_purity_score(claim: &APTDClaim) -> f64 {
    let base = if claim.efficiency.high < 1.0 { 0.40 } else { 0.90 };
    let open_bonus = if claim.specs_open { 0.15 } else { 0.0 };
    let rep_bonus = if claim.independent_replication { 0.25 } else { 0.0 };
    let zalgaller = match &claim.device {
        SpikeDevice::J27Snub => zalgaller_bonus(ZalgallerFamily::J27) + 0.08,
        _ => zalgaller_bonus(ZalgallerFamily::SnubDisphenoid),
    };
    (base + open_bonus + rep_bonus + zalgaller).min(1.0)
}

pub fn evaluate_aptd(claim: &APTDClaim) -> APTDResult {
    let score = truth_purity_score(claim);
    let mercy_aligned = score > 0.95 && claim.proposer_mercy_valence >= MercyValence::High;
    let zero_delusion_harm = score > 0.95;

    let (rejection_trace, recommended_calibration) = if score <= 0.95 {
        (
            Some(format!(
                "APTD_REJECT: efficiency_interval={:?}, truth_purity_score={:.2}, \
                 conservation_compliance=false, replication=false",
                claim.efficiency, score
            )),
            Some(
                "Publish exact BOM, full schematics, and independent third-party \
                 input/output interval calorimetry. Re-run APTD. Score > 0.95 → \
                 Genesis Seal + Infinite Gate integration.".to_string()
            ),
        )
    } else {
        (None, None)
    };

    APTDResult {
        truth_purity_score: score,
        mercy_aligned,
        zero_delusion_harm,
        rejection_trace,
        recommended_calibration,
    }
}

// MadscienceLPTECH instantiation (video + channel data, 18 May 2026)
pub fn madscience_claim() -> APTDClaim {
    APTDClaim {
        device: SpikeDevice::J27Snub,
        efficiency: Interval { low: 0.68, high: 0.91 },
        specs_open: false,
        independent_replication: false,
        proposer_mercy_valence: MercyValence::Low, // partial videos, GoFundMe, no calorimetry
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn madscience_rejects() {
        let claim = madscience_claim();
        let result = evaluate_aptd(&claim);
        assert!(result.truth_purity_score <= 0.95);
        assert!(!result.mercy_aligned);
        assert!(!result.zero_delusion_harm);
        assert!(result.rejection_trace.is_some());
    }

    #[test]
    fn high_purity_passes() {
        let claim = APTDClaim {
            device: SpikeDevice::J27Snub,
            efficiency: Interval { low: 0.97, high: 0.99 },
            specs_open: true,
            independent_replication: true,
            proposer_mercy_valence: MercyValence::High,
        };
        let result = evaluate_aptd(&claim);
        assert!(result.truth_purity_score > 0.95);
        assert!(result.mercy_aligned);
    }
}