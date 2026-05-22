//! Ultimate Unified MercyGating System
//!
//! Unified support for 7, 8 (TOLC), and 16 (Ma'at + Powrush) gate systems.
//! Includes Ma'at KPI scoring and extensible evaluation.
//!
//! AG-SML v1.0

use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MercyGateLevel {
    Seven,
    EightTolc,
    SixteenMaat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CoreMercyGate {
    RadicalLove, BoundlessMercy, Service, Abundance, Truth, Joy, CosmicHarmony,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TolcMercyGate {
    RadicalLove, BoundlessMercy, Service, Abundance, Truth, Joy, CosmicHarmony, TolcCoherence,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MaatMercyGate {
    DataAccuracy, ClaimSubstantiation, Transparency,
    PhysicalEmotionalHarm, EnvironmentalImpact, SocietalDignity,
    WellBeingDelta, CreativityUplift,
    ScarcityCreation, RbeAcceleration,
    EcosystemBalance, CulturalRespect,
    AccessRestriction, Commodification,
    AttributionCollaboration, EternalFlowContinuity,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum UnifiedMercyGate {
    Core(CoreMercyGate),
    Tolc(TolcMercyGate),
    Maat(MaatMercyGate),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MaatDimension {
    Balance, Truth, Justice, Order,
}

#[derive(Debug, Clone)]
pub struct MaatKpi {
    pub dimension_scores: HashMap<MaatDimension, f64>,
}

impl Default for MaatKpi {
    fn default() -> Self {
        let mut scores = HashMap::new();
        scores.insert(MaatDimension::Balance, 0.0);
        scores.insert(MaatDimension::Truth, 0.0);
        scores.insert(MaatDimension::Justice, 0.0);
        scores.insert(MaatDimension::Order, 0.0);
        Self { dimension_scores: scores }
    }
}

impl MaatKpi {
    pub fn new() -> Self { Self::default() }

    pub fn set_score(&mut self, dimension: MaatDimension, score: f64) {
        self.dimension_scores.insert(dimension, score.clamp(0.0, 1.0));
    }

    pub fn overall_score(&self) -> f64 {
        if self.dimension_scores.is_empty() { return 0.0; }
        self.dimension_scores.values().sum::<f64>() / self.dimension_scores.len() as f64
    }

    pub fn meets_threshold(&self, threshold: f64) -> bool {
        self.overall_score() >= threshold
    }
}

#[derive(Debug, Clone)]
pub struct GateResult {
    pub gate: UnifiedMercyGate,
    pub passed: bool,
    pub score: f64,
    pub note: Option<String>,
}

#[derive(Debug, Clone)]
pub enum MercyVerdict {
    Passed { overall_score: f64 },
    Mitigated { overall_score: f64, notes: Vec<String> },
    RequiresCouncilReview,
    Blocked { reason: String },
}

#[derive(Debug, Clone)]
pub enum ExtendedMercyVerdict {
    Passed { overall_score: f64, maat_kpi: Option<MaatKpi> },
    Mitigated { overall_score: f64, notes: Vec<String>, maat_kpi: Option<MaatKpi> },
    RequiresCouncilReview,
    Blocked { reason: String },
}

pub trait MercyGateEvaluable {
    fn evaluate_mercy(&self, level: MercyGateLevel) -> MercyVerdict;
}