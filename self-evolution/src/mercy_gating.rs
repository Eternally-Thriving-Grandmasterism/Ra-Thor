//! # Ultimate Unified MercyGating System (Deep Implementation v0.6.0)
//!
//! Expanded to support a rich, coherent set of Mercy Gates.
//! Primary layers: Foundational 7 + Operational 16 (Ma'at) + Integrative/Meta gates.
//!
//! New gates added for depth and usefulness (non-diminishing):
//! - PATSAGi Consensus Gate
//! - Self-Evolution Blessing Gate
//! - Lattice Coherence Gate
//! - TOLC Fidelity Gate
//! - ONE Organism Symbiosis Gate
//! - Quantum Swarm Mercy Gate
//! - Genesis / Origin Gate

use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MercyGateLevel {
    Foundational,      // 7 Living Mercy Gates
    Operational,       // 16-gate Ma'at system
    Integrative,       // New meta / integration gates
}

// Foundational 7
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FoundationalMercyGate {
    RadicalLove,
    BoundlessMercy,
    Service,
    Abundance,
    Truth,
    Joy,
    CosmicHarmony,
}

// Operational 16 (Ma'at focused)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperationalMercyGate {
    DataAccuracy, ClaimSubstantiation, Transparency,
    PhysicalEmotionalHarm, EnvironmentalImpact, SocietalDignity,
    WellBeingDelta, CreativityUplift,
    ScarcityCreation, RbeAcceleration,
    EcosystemBalance, CulturalRespect,
    AccessRestriction, Commodification,
    AttributionCollaboration, EternalFlowContinuity,
}

// New Integrative / Meta Gates (proposed expansion)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IntegrativeMercyGate {
    PatsagiConsensus,
    SelfEvolutionBlessing,
    LatticeCoherence,
    TolcFidelity,
    OneOrganismSymbiosis,
    QuantumSwarmMercy,
    GenesisOrigin,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum UnifiedMercyGate {
    Foundational(FoundationalMercyGate),
    Operational(OperationalMercyGate),
    Integrative(IntegrativeMercyGate),
}

// Ma'at dimensions remain
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MaatDimension {
    Balance, Truth, Justice, Order,
}

#[derive(Debug, Clone)]
pub struct MaatKpi {
    pub dimension_scores: HashMap<MaatDimension, f64>,
}

// ... (impls for MaatKpi remain similar)

#[derive(Debug, Clone)]
pub enum MercyVerdict {
    Passed { overall_score: f64 },
    Mitigated { overall_score: f64, notes: Vec<String> },
    RequiresCouncilReview,
    Blocked { reason: String },
}

pub trait MercyGateEvaluable {
    fn evaluate_mercy(&self, level: MercyGateLevel) -> MercyVerdict;
}

// Implementation for SnapshotError with enriched logic
impl MercyGateEvaluable for crate::SnapshotError {
    fn evaluate_mercy(&self, level: MercyGateLevel) -> MercyVerdict {
        let base_score = match self {
            crate::SnapshotError::FileNotFound { .. } => 0.82,
            crate::SnapshotError::ReadError { .. } => 0.78,
            crate::SnapshotError::ParseError { .. } => 0.65,
            crate::SnapshotError::UnknownFormat => 0.60,
        };

        match level {
            MercyGateLevel::Foundational => {
                if base_score >= 0.78 {
                    MercyVerdict::Mitigated { overall_score: base_score, notes: vec!["Foundational mercy evaluation".to_string()] }
                } else {
                    MercyVerdict::RequiresCouncilReview }
            }
            MercyGateLevel::Operational => {
                // Rich Ma'at scoring
                let mut kpi = MaatKpi::new();
                kpi.set_score(MaatDimension::Truth, base_score * 0.96);
                kpi.set_score(MaatDimension::Balance, base_score * 0.92);
                kpi.set_score(MaatDimension::Justice, base_score * 0.87);
                kpi.set_score(MaatDimension::Order, base_score * 0.90);

                let score = kpi.overall_score();
                if score >= 0.88 { MercyVerdict::Passed { overall_score: score } }
                else if score >= 0.72 { MercyVerdict::Mitigated { overall_score: score, notes: vec![format!("Ma'at: {:.3}", score)] } }
                else { MercyVerdict::RequiresCouncilReview }
            }
            MercyGateLevel::Integrative => {
                // For now, route integrative gates through strong Operational logic + council review bias
                if base_score >= 0.85 {
                    MercyVerdict::Mitigated { overall_score: base_score, notes: vec!["Integrative layer evaluation".to_string()] }
                } else {
                    MercyVerdict::RequiresCouncilReview }
            }
        }
    }
}