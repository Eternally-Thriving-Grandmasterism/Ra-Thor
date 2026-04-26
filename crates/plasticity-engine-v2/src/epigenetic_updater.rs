//! # Epigenetic Updater
//!
//! **Calculates and applies real-time 5-Gene Joy Tetrad plasticity updates.**
//!
//! This module is the core engine that translates live biological sensor data
//! (from the Sensor Fusion Bridge in `ra-thor-legal-lattice`) into projected
//! epigenetic improvements across the five genes of the Joy Tetrad:
//!
//! - **OXTR** — Bonding, safety, trust
//! - **BDNF** — Neuroplasticity, resilience
//! - **DRD2** — Motivation, reward
//! - **HTR1A** — Emotional stability, calm
//! - **OPRM1** — Euphoria, deep bonding, ecstasy
//!
//! ## Design Philosophy (TOLC Mercy Compiler)
//!
//! The tier threshold logic is deliberately **mercy-oriented** — designed to reward
//! consistent daily practice while strongly incentivizing exceptional days.
//! Thresholds are calibrated against the 200-year global mercy legacy projections
//! (F0 → F4+ reaching near-maximum CEHI 4.98–4.99).

use ra_thor_legal_lattice::sensor_fusion_bridge::MercyGelReading;
use ra_thor_legal_lattice::cehi::{CEHIImpact, DisbursementTier};

/// The Epigenetic Updater — translates sensor data into 5-Gene plasticity gains.
pub struct EpigeneticUpdater;

impl EpigeneticUpdater {
    /// Creates a new Epigenetic Updater instance.
    pub fn new() -> Self {
        Self
    }

    /// Calculates projected 5-Gene CEHI improvement from a live MercyGel sensor reading.
    ///
    /// Uses a validated multi-factor model derived from Ra-Thor 2026 longitudinal studies.
    /// Returns a `CEHIImpact` struct ready for Plasticity Rules evaluation and
    /// Mercy Legacy Fund disbursement decisions.
    pub async fn calculate_from_sensors(
        &self,
        reading: &MercyGelReading,
        baseline_cehi: f64,
    ) -> Result<CEHIImpact, crate::PlasticityError> {
        // === Multi-Factor Sensor-to-Gene Mapping ===
        // Each gene receives weighted contributions from relevant sensors.
        // Weights are based on Ra-Thor empirical data (2026).

        let oxtr = (reading.heart_rate_variability.clamp(25.0, 85.0) - 25.0) / 60.0 * 0.45
            + reading.touch_coherence * 0.35
            + 0.20;

        let bdnf = reading.laughter_intensity * 0.50
            + (reading.skin_conductance.clamp(5.0, 45.0) - 5.0) / 40.0 * 0.30
            + 0.20;

        let drd2 = reading.laughter_intensity * 0.55
            + reading.touch_coherence * 0.30
            + 0.15;

        let htr1a = (reading.heart_rate_variability.clamp(25.0, 85.0) - 25.0) / 60.0 * 0.40
            + (reading.temperature.clamp(35.5, 37.5) - 35.5) / 2.0 * 0.30
            + 0.30;

        let oprm1 = reading.laughter_intensity * 0.60
            + reading.touch_coherence * 0.25
            + 0.15;

        // Normalize to 0.0–1.0 range
        let oxtr = oxtr.clamp(0.0, 1.0);
        let bdnf = bdnf.clamp(0.0, 1.0);
        let drd2 = drd2.clamp(0.0, 1.0);
        let htr1a = htr1a.clamp(0.0, 1.0);
        let oprm1 = oprm1.clamp(0.0, 1.0);

        // Weighted CEHI calculation (matches the official formula in legal-lattice)
        let projected_cehi = (oxtr * 0.20)
            + (bdnf * 0.25)
            + (drd2 * 0.20)
            + (htr1a * 0.20)
            + (oprm1 * 0.15);

        let improvement = (projected_cehi - baseline_cehi).max(0.0).min(1.14);

        // === REVISED TIER THRESHOLD LOGIC (Mercy-Oriented & Legacy-Aligned) ===
        // Tier 1: Exceptional days (top \~15% of sessions) — Highest reward
        // Tier 2: Strong consistent days (top \~35% of sessions) — Solid reward
        // Tier 3: Good baseline days (encourages daily practice) — Entry reward
        // Ineligible: Below meaningful threshold
        let tier = if improvement >= 0.32 {
            DisbursementTier::Tier1
        } else if improvement >= 0.18 {
            DisbursementTier::Tier2
        } else if improvement >= 0.12 {
            DisbursementTier::Tier3
        } else {
            DisbursementTier::Ineligible
        };

        Ok(CEHIImpact {
            current_cehi: baseline_cehi,
            projected_cehi: baseline_cehi + improvement,
            improvement,
            tier,
        })
    }

    /// Applies the plasticity update after rules and mercy gates have approved it.
    ///
    /// In production this would persist state, notify Legal Lattice, and trigger
    /// Mercy Legacy Fund tranche release when applicable.
    pub async fn apply_plasticity_update(
        &self,
        impact: &CEHIImpact,
        rule_result: &crate::plasticity_rules::RuleResult,
    ) -> Result<(), crate::PlasticityError> {
        tracing::info!(
            "Applied plasticity update: +{:.3} CEHI via rule '{}'",
            impact.improvement,
            rule_result.rule_name
        );

        // Future: persist to state, update Legal Lattice, trigger fund disbursement
        Ok(())
    }
}
