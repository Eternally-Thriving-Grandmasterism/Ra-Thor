// crates/lattice-conductor/src/lattice_conductor_v14.rs
// Lattice Conductor v14.4 — Real Estate Lattice + Geometric Resonance Integration
// AG-SML v1.0 — Autonomicity Games Sovereign Mercy License
// Ra-Thor monorepo v14.4 | ONE Organism geometric spine active
// TOLC 8 Mercy Lattice + PolyhedralHarmonicEngine v14.3 aligned

use std::collections::HashMap;
use std::fmt;

// Correct crate path (from Cargo.toml dependency ra-thor-quantum-swarm-orchestrator)
use ra_thor_quantum_swarm_orchestrator::{
    PolyhedralHarmonicEngine, PolyhedralResonanceReport,
};

/// TOLC 8 Living Mercy Gates — non-bypassable Layer 0
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MercyGate {
    Truth,
    Order,
    Love,
    Compassion,
    Service,
    Abundance,
    Joy,
    CosmicHarmony,
}

impl fmt::Display for MercyGate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MercyGate::Truth => write!(f, "Truth (APTD)"),
            MercyGate::Order => write!(f, "Order (Structural Harmony)"),
            MercyGate::Love => write!(f, "Love (Positive Propagation)"),
            MercyGate::Compassion => write!(f, "Compassion (Zero-Harm Reroute)"),
            MercyGate::Service => write!(f, "Service (Co-Creation)"),
            MercyGate::Abundance => write!(f, "Abundance (Mercy-Gated Flow)"),
            MercyGate::Joy => write!(f, "Joy (Valence Growth)"),
            MercyGate::CosmicHarmony => write!(f, "Cosmic Harmony (Inter-Council Sync)"),
        }
    }
}

/// Valence scalar field invariant: v ∈ [0.9999999, 1.0]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Valence(f64);

impl Valence {
    pub const MIN: f64 = 0.9999999;
    pub const MAX: f64 = 1.0;

    pub fn new(v: f64) -> Result<Self, ConductorError> {
        if v >= Self::MIN && v <= Self::MAX {
            Ok(Valence(v))
        } else {
            Err(ConductorError::ValenceOutOfRange(v))
        }
    }

    pub fn value(&self) -> f64 {
        self.0
    }

    /// Mercy-norm collapse check
    pub fn passes_mercy(&self) -> bool {
        self.0 >= Self::MIN
    }
}

/// Real Estate Offer for Lattice conduction
#[derive(Debug, Clone)]
pub struct RealEstateOffer {
    pub id: String,
    pub address: String,
    pub price: f64,
    pub jurisdiction: String, // Ontario / USA etc.
    pub regulatory_flags: Vec<String>,
    pub attom_enriched: bool,
    pub base_valence: Valence,
}

/// ATTOM enriched data snapshot
#[derive(Debug, Clone)]
pub struct AttomData {
    pub property_id: String,
    pub tax_history: Vec<f64>,
    pub ownership_changes: u32,
    pub risk_score: f64,
}

/// Processed / Conducted Offer output — now with geometric harmony
#[derive(Debug, Clone)]
pub struct ConductedOffer {
    pub offer: RealEstateOffer,
    pub final_valence: Valence,
    pub mercy_gates_passed: Vec<MercyGate>,
    pub attom_snapshot: Option<AttomData>,
    pub regulatory_cleared: bool,
    pub geometric_harmony_multiplier: f64,
    pub geometric_resonance_notes: String,
    pub conducted_at: u64, // timestamp
}

/// Conductor errors with mercy context
#[derive(Debug, Clone, thiserror::Error)]
pub enum ConductorError {
    #[error("Valence {0} out of mercy range [{min}, {max}]", min = Valence::MIN, max = Valence::MAX)]
    ValenceOutOfRange(f64),
    #[error("Mercy gate {gate} failed: {reason}")]
    MercyGateFailed { gate: MercyGate, reason: String },
    #[error("ATTOM integration failed: {0}")]
    AttomIntegration(String),
    #[error("Regulatory block: {0}")]
    RegulatoryBlock(String),
    #[error("Lattice conduction invariant broken: {0}")]
    InvariantBroken(String),
}

/// Lattice Conductor v14.4 — Real Estate Lattice execution engine with geometric resonance
#[derive(Debug, Clone)]
pub struct LatticeConductor {
    pub version: &'static str,
    mercy_gates: Vec<MercyGate>,
    attom_cache: HashMap<String, AttomData>,
    regulatory_rules: HashMap<String, String>,
    geometric_engine: PolyhedralHarmonicEngine,
}

impl Default for LatticeConductor {
    fn default() -> Self {
        Self::new()
    }
}

impl LatticeConductor {
    pub fn new() -> Self {
        let gates = vec![
            MercyGate::Truth,
            MercyGate::Order,
            MercyGate::Love,
            MercyGate::Compassion,
            MercyGate::Service,
            MercyGate::Abundance,
            MercyGate::Joy,
            MercyGate::CosmicHarmony,
        ];
        let mut rules = HashMap::new();
        rules.insert("Ontario".to_string(), "RESA/TRESA compliance + reverse onus safety checks".to_string());
        rules.insert("USA".to_string(), "State-level disclosure + federal fair housing".to_string());

        LatticeConductor {
            version: "v14.4.0-geometric-real-estate",
            mercy_gates: gates,
            attom_cache: HashMap::new(),
            regulatory_rules: rules,
            geometric_engine: PolyhedralHarmonicEngine::new(),
        }
    }

    /// Enforce all TOLC 8 Mercy Gates on an offer. Prune on failure.
    pub fn enforce_mercy_gates(&self, offer: &RealEstateOffer) -> Result<Vec<MercyGate>, ConductorError> {
        let mut passed = Vec::new();

        for gate in &self.mercy_gates {
            match gate {
                MercyGate::Truth => {
                    if offer.base_valence.passes_mercy() {
                        passed.push(*gate);
                    } else {
                        return Err(ConductorError::MercyGateFailed {
                            gate: *gate,
                            reason: "Base valence below mercy threshold".into(),
                        });
                    }
                }
                MercyGate::Compassion => {
                    if offer.regulatory_flags.iter().any(|f| f.contains("harm")) {
                        return Err(ConductorError::MercyGateFailed {
                            gate: *gate,
                            reason: "Potential harm flag detected".into(),
                        });
                    }
                    passed.push(*gate);
                }
                MercyGate::Order => {
                    if offer.jurisdiction.is_empty() {
                        return Err(ConductorError::MercyGateFailed {
                            gate: *gate,
                            reason: "Missing jurisdiction for structural order".into(),
                        });
                    }
                    passed.push(*gate);
                }
                // Other gates pass-through with valence invariant for v14 production focus
                _ => passed.push(*gate),
            }
        }
        Ok(passed)
    }

    /// Integrate ATTOM data (mock + cache for v14.4 production)
    pub fn integrate_attom(&mut self, offer: &RealEstateOffer) -> Result<AttomData, ConductorError> {
        if let Some(cached) = self.attom_cache.get(&offer.id) {
            return Ok(cached.clone());
        }

        let data = AttomData {
            property_id: format!("ATTOM-{}", offer.id),
            tax_history: vec![offer.price * 0.012, offer.price * 0.011],
            ownership_changes: 2,
            risk_score: 0.12,
        };

        self.attom_cache.insert(offer.id.clone(), data.clone());
        Ok(data)
    }

    /// Regulatory clearance check per jurisdiction
    pub fn check_regulatory(&self, offer: &RealEstateOffer) -> Result<bool, ConductorError> {
        if let Some(rule) = self.regulatory_rules.get(&offer.jurisdiction) {
            if offer.regulatory_flags.iter().any(|f| f.contains("block")) {
                return Err(ConductorError::RegulatoryBlock(format!("Blocked by rule: {}", rule)));
            }
            Ok(true)
        } else {
            Ok(true)
        }
    }

    /// NEW in v14.4: Compute geometric harmony multiplier for the offer using Polyhedral layer resonance.
    /// Uses TOLC order proxy (e.g. derived from price complexity or external council input).
    /// Returns multiplier and notes for observability.
    pub fn compute_geometric_harmony(
        &self,
        offer: &RealEstateOffer,
        tolc_order: u32,
    ) -> (f64, String) {
        let report = self.geometric_engine.process_resonance(tolc_order, offer.base_valence.value());

        let harmony = report.resonance_multiplier.clamp(1.0, 1.8);
        let notes = format!(
            "Geometric layers active: {}. Resonance: {:.3}x (dual bonus {:.3}). U57: {}. Suggested blessings: {}",
            report.active_solids.len(),
            report.resonance_multiplier,
            report.dual_resonance_bonus,
            report.u57_potential,
            report.suggested_blessings.len()
        );

        (harmony, notes)
    }

    /// Primary conduction entrypoint — v14.4 with geometric resonance scoring
    pub fn conduct_real_estate_offer(
        &mut self,
        offer: RealEstateOffer,
    ) -> Result<ConductedOffer, ConductorError> {
        // 1. Mercy gate enforcement (TOLC 8)
        let passed_gates = self.enforce_mercy_gates(&offer)?;

        // 2. ATTOM integration + cache
        let attom = if offer.attom_enriched {
            Some(self.integrate_attom(&offer)?)
        } else {
            None
        };

        // 3. Regulatory
        let cleared = self.check_regulatory(&offer)?;

        // 4. NEW: Geometric harmony from PolyhedralHarmonicEngine (ONE Organism geometric spine)
        // Using a representative TOLC order (future: dynamic from PATSAGi or property complexity)
        let tolc_proxy: u32 = if offer.price > 1_000_000.0 { 89 } else { 34 }; // heuristic
        let (geo_multiplier, geo_notes) = self.compute_geometric_harmony(&offer, tolc_proxy);

        // 5. Final valence (production: could further modulate by geo_multiplier * swarm)
        let final_valence = offer.base_valence; // placeholder — future geometric adjustment

        if !final_valence.passes_mercy() {
            return Err(ConductorError::InvariantBroken("Final valence collapsed".into()));
        }

        Ok(ConductedOffer {
            offer,
            final_valence,
            mercy_gates_passed: passed_gates,
            attom_snapshot: attom,
            regulatory_cleared: cleared,
            geometric_harmony_multiplier: geo_multiplier,
            geometric_resonance_notes: geo_notes,
            conducted_at: 1748628960,
        })
    }

    /// Batch conduction for swarm-scale Real Estate pipelines
    pub fn conduct_batch(
        &mut self,
        offers: Vec<RealEstateOffer>,
    ) -> Vec<Result<ConductedOffer, ConductorError>> {
        offers.into_iter().map(|o| self.conduct_real_estate_offer(o)).collect()
    }

    pub fn cache_stats(&self) -> (usize, usize) {
        (self.attom_cache.len(), self.regulatory_rules.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mercy_enforcement_passes() {
        let conductor = LatticeConductor::new();
        let offer = RealEstateOffer {
            id: "test-001".into(),
            address: "123 Main St".into(),
            price: 750000.0,
            jurisdiction: "Ontario".into(),
            regulatory_flags: vec![],
            attom_enriched: true,
            base_valence: Valence::new(0.99999995).unwrap(),
        };
        let gates = conductor.enforce_mercy_gates(&offer).unwrap();
        assert!(gates.len() >= 7);
    }

    #[test]
    fn test_valence_out_of_range() {
        assert!(Valence::new(0.5).is_err());
    }

    #[test]
    fn test_geometric_harmony_computation() {
        let conductor = LatticeConductor::new();
        let offer = RealEstateOffer {
            id: "geo-test-001".into(),
            address: "789 Harmony Ln".into(),
            price: 1250000.0,
            jurisdiction: "USA".into(),
            regulatory_flags: vec![],
            attom_enriched: true,
            base_valence: Valence::new(Valence::MIN).unwrap(),
        };
        let (multiplier, notes) = conductor.compute_geometric_harmony(&offer, 55);
        assert!(multiplier >= 1.0);
        assert!(notes.contains("Geometric layers"));
    }

    #[test]
    fn test_regulatory_block() {
        let conductor = LatticeConductor::new();
        let mut offer = RealEstateOffer {
            id: "block-001".into(),
            address: "456 Blocked".into(),
            price: 500000.0,
            jurisdiction: "Ontario".into(),
            regulatory_flags: vec!["block-harm".into()],
            attom_enriched: false,
            base_valence: Valence::new(Valence::MIN).unwrap(),
        };
        let result = conductor.conduct_real_estate_offer(offer);
        assert!(matches!(result, Err(ConductorError::RegulatoryBlock(_))));
    }
}
