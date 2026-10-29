//! Energy design proposals scored through TOLC 8 Live Valence floors.
//!
//! Maps explicit design metrics → `ValenceVector` / `LiveValenceReport`.
//! Same soft (0.55) and strict (0.72) floors as telemetry valence.
//! Does **not** alter Cosmic Tick state (quiet-hold compliant).
//!
//! AG-SML v1.0 | Contact: info@Rathor.ai

use serde::{Deserialize, Serialize};

use crate::live_valence::{
    LiveValenceReport, ValenceVector, THETA_MIN_SOFT, THETA_MIN_STRICT,
};

// =============================================================================
// Proposal types
// =============================================================================

/// Class of energy system under evaluation (design-time, not deployment claim).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EnergyDesignClass {
    /// Small modular reactor (terrestrial).
    Smr,
    /// Advanced geothermal / closed-loop heat.
    Geothermal,
    /// Fusion-adjacent or magnetic-confinement concept.
    FusionAdjacent,
    /// Helium-3 / lunar-fuel conceptual pathway.
    He3Pathway,
    /// Distributed solar + storage abundance stack.
    SolarAbundance,
    /// Hybrid multi-source lattice.
    HybridLattice,
    /// Other / experimental (must still pass floors).
    Experimental,
}

impl EnergyDesignClass {
    pub fn as_str(&self) -> &'static str {
        match self {
            EnergyDesignClass::Smr => "SMR",
            EnergyDesignClass::Geothermal => "Geothermal",
            EnergyDesignClass::FusionAdjacent => "FusionAdjacent",
            EnergyDesignClass::He3Pathway => "He3Pathway",
            EnergyDesignClass::SolarAbundance => "SolarAbundance",
            EnergyDesignClass::HybridLattice => "HybridLattice",
            EnergyDesignClass::Experimental => "Experimental",
        }
    }
}

/// Explicit, auditable metrics for an energy design proposal.
/// Every field is a [0,1] score unless noted. Higher = better alignment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergyDesignProposal {
    pub id: String,
    pub title: String,
    pub class: EnergyDesignClass,
    /// Truth: evidence grounding, falsifiability, open data quality.
    pub evidence_grounding: f64,
    /// Order: control stability, predictability, formal safety cases.
    pub control_stability: f64,
    /// Love: shared access, community benefit, non-exclusionary design.
    pub community_benefit: f64,
    /// Compassion / Zero-Harm: passive safety, failure modes, externalities.
    pub zero_harm_safety: f64,
    /// Service: open protocols, transferability, local operability.
    pub open_serviceability: f64,
    /// Abundance: capacity density, cost trajectory, scalability proxy.
    pub abundance_density: f64,
    /// Joy: livability impact, local air/water quality, human flourishing.
    pub livability_impact: f64,
    /// Cosmic Harmony: multi-generational + multi-planetary resilience.
    pub long_horizon_harmony: f64,
    /// Optional free-text design notes (not scored).
    #[serde(default)]
    pub notes: String,
}

impl EnergyDesignProposal {
    /// Validate all scored fields are in [0, 1].
    pub fn validate_bounds(&self) -> Result<(), String> {
        let fields = [
            ("evidence_grounding", self.evidence_grounding),
            ("control_stability", self.control_stability),
            ("community_benefit", self.community_benefit),
            ("zero_harm_safety", self.zero_harm_safety),
            ("open_serviceability", self.open_serviceability),
            ("abundance_density", self.abundance_density),
            ("livability_impact", self.livability_impact),
            ("long_horizon_harmony", self.long_horizon_harmony),
        ];
        for (name, v) in fields {
            if !(0.0..=1.0).contains(&v) {
                return Err(format!(
                    "Mercy Gate (Truth): energy design field '{}' out of [0,1] (got {})",
                    name, v
                ));
            }
        }
        if self.id.trim().is_empty() || self.title.trim().is_empty() {
            return Err("Mercy Gate (Truth): energy design id/title must be non-empty".into());
        }
        Ok(())
    }
}

// =============================================================================
// Scoring
// =============================================================================

/// Full energy design score package (valence + design identity).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergyDesignScore {
    pub proposal_id: String,
    pub title: String,
    pub class: String,
    pub valence: LiveValenceReport,
    /// Council-facing recommendation string.
    pub recommendation: String,
}

/// Map proposal metrics → ValenceVector (1:1 gate alignment).
fn map_proposal_to_vector(p: &EnergyDesignProposal) -> ValenceVector {
    ValenceVector {
        truth: p.evidence_grounding.clamp(0.0, 1.0),
        order: p.control_stability.clamp(0.0, 1.0),
        love: p.community_benefit.clamp(0.0, 1.0),
        compassion: p.zero_harm_safety.clamp(0.0, 1.0),
        service: p.open_serviceability.clamp(0.0, 1.0),
        abundance: p.abundance_density.clamp(0.0, 1.0),
        joy: p.livability_impact.clamp(0.0, 1.0),
        cosmic_harmony: p.long_horizon_harmony.clamp(0.0, 1.0),
    }
}

fn report_from_vector(vector: ValenceVector) -> LiveValenceReport {
    let min_gate = vector.min_gate();
    let aggregate_mean = vector.mean();
    let passes_soft = vector.passes_soft_floor();
    let passes_strict = vector.passes_strict_floor();

    let council_note = if passes_strict {
        format!(
            "Energy design STRICT PASS | min={:.3} mean={:.3} | all TOLC 8 ≥ {:.2}",
            min_gate, aggregate_mean, THETA_MIN_STRICT
        )
    } else if passes_soft {
        format!(
            "Energy design SOFT PASS | min={:.3} mean={:.3} | below strict {:.2} — council review",
            min_gate, aggregate_mean, THETA_MIN_STRICT
        )
    } else {
        format!(
            "Energy design FLOOR FAIL | min={:.3} mean={:.3} | soft floor {:.2} not met — hold",
            min_gate, aggregate_mean, THETA_MIN_SOFT
        )
    };

    LiveValenceReport {
        vector,
        aggregate_mean,
        min_gate,
        passes_soft_floor: passes_soft,
        passes_strict_floor: passes_strict,
        council_note,
    }
}

/// Score an energy design proposal under TOLC 8 Live Valence floors.
pub fn score_energy_design(proposal: &EnergyDesignProposal) -> Result<EnergyDesignScore, String> {
    proposal.validate_bounds()?;
    let vector = map_proposal_to_vector(proposal);
    let valence = report_from_vector(vector);

    let recommendation = if valence.passes_strict_floor {
        format!(
            "ADVANCE: '{}' ({}) clears strict TOLC 8. Open shard + formal safety case next.",
            proposal.title,
            proposal.class.as_str()
        )
    } else if valence.passes_soft_floor {
        format!(
            "REVIEW: '{}' ({}) soft-pass only (min={:.3}). Raise weakest gates before hardware path.",
            proposal.title,
            proposal.class.as_str(),
            valence.min_gate
        )
    } else {
        format!(
            "HOLD: '{}' ({}) fails soft floor (min={:.3}). Redesign under Zero-Harm + Truth before cascade.",
            proposal.title,
            proposal.class.as_str(),
            valence.min_gate
        )
    };

    Ok(EnergyDesignScore {
        proposal_id: proposal.id.clone(),
        title: proposal.title.clone(),
        class: proposal.class.as_str().into(),
        valence,
        recommendation,
    })
}

// =============================================================================
// Example proposals (design-time illustrations, not claims of physical readiness)
// =============================================================================

/// High-valence open SMR-style concept oriented to abundance + passive safety.
pub fn example_open_smr_high() -> EnergyDesignProposal {
    EnergyDesignProposal {
        id: "energy-open-smr-001".into(),
        title: "Open Passive-Safety SMR Lattice (abundance-first)".into(),
        class: EnergyDesignClass::Smr,
        evidence_grounding: 0.88,
        control_stability: 0.86,
        community_benefit: 0.84,
        zero_harm_safety: 0.91,
        open_serviceability: 0.89,
        abundance_density: 0.87,
        livability_impact: 0.83,
        long_horizon_harmony: 0.85,
        notes: "Design-time open protocol + passive safety emphasis; not a licensed reactor claim."
            .into(),
    }
}

/// Marginal geothermal / hybrid — soft-pass territory.
pub fn example_geothermal_marginal() -> EnergyDesignProposal {
    EnergyDesignProposal {
        id: "energy-geo-marginal-002".into(),
        title: "Closed-Loop Geothermal Node (early evidence)".into(),
        class: EnergyDesignClass::Geothermal,
        evidence_grounding: 0.62,
        control_stability: 0.70,
        community_benefit: 0.68,
        zero_harm_safety: 0.74,
        open_serviceability: 0.66,
        abundance_density: 0.58,
        livability_impact: 0.71,
        long_horizon_harmony: 0.64,
        notes: "Early-stage; abundance density and evidence need lift for strict floor.".into(),
    }
}

/// Fails soft floor — low zero-harm + weak evidence.
pub fn example_experimental_fail() -> EnergyDesignProposal {
    EnergyDesignProposal {
        id: "energy-exp-fail-003".into(),
        title: "Unverified Exotic Propulsion Heat Source".into(),
        class: EnergyDesignClass::Experimental,
        evidence_grounding: 0.35,
        control_stability: 0.40,
        community_benefit: 0.50,
        zero_harm_safety: 0.28,
        open_serviceability: 0.45,
        abundance_density: 0.60,
        livability_impact: 0.42,
        long_horizon_harmony: 0.38,
        notes: "Intentionally weak Zero-Harm + Truth to exercise HOLD path.".into(),
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn high_smr_strict_pass() {
        let score = score_energy_design(&example_open_smr_high()).unwrap();
        assert!(score.valence.passes_strict_floor);
        assert!(score.valence.passes_soft_floor);
        assert!(score.valence.min_gate >= THETA_MIN_STRICT);
        assert!(score.recommendation.starts_with("ADVANCE"));
        assert_eq!(score.class, "SMR");
    }

    #[test]
    fn geothermal_soft_or_borderline() {
        let score = score_energy_design(&example_geothermal_marginal()).unwrap();
        assert!(score.valence.passes_soft_floor);
        // marginal abundance keeps it below strict in the example
        assert!(!score.valence.passes_strict_floor);
        assert!(score.recommendation.starts_with("REVIEW"));
    }

    #[test]
    fn experimental_floor_fail() {
        let score = score_energy_design(&example_experimental_fail()).unwrap();
        assert!(!score.valence.passes_soft_floor);
        assert!(score.valence.min_gate < THETA_MIN_SOFT);
        assert!(score.recommendation.starts_with("HOLD"));
        // Compassion should be the weak gate in this fixture
        assert!(score.valence.vector.compassion < THETA_MIN_SOFT);
    }

    #[test]
    fn rejects_out_of_bounds() {
        let mut p = example_open_smr_high();
        p.zero_harm_safety = 1.2;
        assert!(score_energy_design(&p).is_err());
    }

    #[test]
    fn rejects_empty_id() {
        let mut p = example_open_smr_high();
        p.id = "   ".into();
        assert!(score_energy_design(&p).is_err());
    }

    #[test]
    fn vector_gates_match_proposal_fields() {
        let p = example_open_smr_high();
        let score = score_energy_design(&p).unwrap();
        assert!((score.valence.vector.truth - p.evidence_grounding).abs() < 1e-12);
        assert!((score.valence.vector.compassion - p.zero_harm_safety).abs() < 1e-12);
        assert!((score.valence.vector.abundance - p.abundance_density).abs() < 1e-12);
    }
}
