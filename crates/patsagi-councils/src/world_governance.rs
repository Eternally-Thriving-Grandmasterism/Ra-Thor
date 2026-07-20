//! # WorldGovernanceEngine — v14.15.0
//!
//! Living heart of Powrush-MMO world governance under the 16 PATSAGi Councils.
//! Faction harmony, economy, diplomacy, culture, quantum mercy fields,
//! and TOLC consultation — mercy-gated at every layer.
//!
//! Living Cosmic Tick aligned. Permanent deliberation posture.
//! Contact: info@Rathor.ai
//! AG-SML v1.0

use crate::tolc_integration::TOLCCouncilBridge;
use crate::CouncilFocus;
use chrono::{DateTime, Utc};
use mercy::MercyEngine;
use powrush::{AscensionLevel, Faction, MercyGateStatus, PowrushGame, ResourceType};
use quantum_swarm_orchestrator::QuantumSwarmOrchestrator;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;
use uuid::Uuid;

pub const VERSION: &str = "14.15.0";

// =============================================================================
// Core proposal / economy types
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldChangeProposal {
    pub id: Uuid,
    pub proposed_by: CouncilFocus,
    pub title: String,
    pub description: String,
    pub impact_type: WorldImpactType,
    pub mercy_cost: f64,
    pub joy_boost: f32,
    pub cehi_boost: f64,
    pub nectar_amount: f64,
    pub timestamp: DateTime<Utc>,
    pub expires_at_cycle: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmbrosianNectarEconomy {
    pub total_nectar: f64,
    pub bloom_count: u64,
    pub last_bloom: Option<DateTime<Utc>>,
}

impl AmbrosianNectarEconomy {
    pub fn new() -> Self {
        Self {
            total_nectar: 10_000.0,
            bloom_count: 0,
            last_bloom: None,
        }
    }

    pub fn bloom(&mut self, amount: f64) {
        self.total_nectar += amount;
        self.bloom_count = self.bloom_count.saturating_add(1);
        self.last_bloom = Some(Utc::now());
    }
}

impl Default for AmbrosianNectarEconomy {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum WorldImpactType {
    FactionAIStrategies,
    FactionTreatySigned,
    AllianceFormed,
    EspionageOperation,
    CulturalFestival,
    PMS_TenantApplicationApproved,
    PMS_MaintenanceRequestResolved,
    PMS_RentAdjustmentHarmonyBoost,
    PMS_LeaseRenewalWithMercy,
    PMS_CommunityRuleUpdate,
    PMS_EvictionPreventionViaMercy,
    USA_RespaViolationPrevented,
    USA_TilaDisclosureGenerated,
    USA_FairHousingViolationPrevented,
    USA_CfpbMortgageApproved,
    USA_EcoaViolationPrevented,
    USA_CaliforniaWildfireDisclosureGenerated,
    USA_FloridaFloodZoneRiskAssessed,
    USA_TexasPropertyTaxAppealGenerated,
    USA_NewYorkRentStabilizationVerified,
    USA_NewJerseyCoastalRiskAssessed,
    USA_GeneralRegulatoryComplianceAchieved,
}

#[derive(Debug, Error)]
pub enum PmsError {
    #[error("Ra-Thor lattice rejected action: mercy valence {valence:.2} < threshold {threshold:.2}")]
    LatticeRejection { valence: f64, threshold: f64 },
    #[error("Quantum swarm consensus too low: {consensus:.1}%")]
    SwarmConsensusTooLow { consensus: f64 },
    #[error("PMS API error: {0}")]
    PmsApiError(String),
    #[error("Validation error: {0}")]
    Validation(String),
    #[error("Human review required: {reason}")]
    HumanReviewRequired { reason: String },
}

// =============================================================================
// Faction Harmony
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactionHarmonyMatrix {
    pub harmony_scores: HashMap<Faction, f64>,
    pub tension_levels: HashMap<Faction, f64>,
    pub synergy_bonus: f64,
    pub war_risk: f64,
    pub last_peace_treaty: Option<DateTime<Utc>>,
    pub harmony_decay_rate: f64,
    pub mercy_influence_multiplier: f64,
}

impl FactionHarmonyMatrix {
    pub fn new() -> Self {
        let mut harmony = HashMap::new();
        let mut tension = HashMap::new();
        for faction in [
            Faction::HarmonyWeavers,
            Faction::TruthSeekers,
            Faction::AbundanceSeekers,
            Faction::AscensionPath,
        ] {
            harmony.insert(faction, 0.72);
            tension.insert(faction, 0.18);
        }
        Self {
            harmony_scores: harmony,
            tension_levels: tension,
            synergy_bonus: 1.0,
            war_risk: 0.12,
            last_peace_treaty: None,
            harmony_decay_rate: 0.008,
            mercy_influence_multiplier: 1.35,
        }
    }

    pub fn boost_harmony(&mut self, faction: Faction, amount: f64, mercy_valence: f64) {
        if let Some(score) = self.harmony_scores.get_mut(&faction) {
            let mercy_boost = mercy_valence * self.mercy_influence_multiplier * 0.1;
            *score = (*score + amount + mercy_boost).min(1.0);
        }
        self.recalculate_all();
    }

    pub fn reduce_tension(&mut self, faction: Faction, amount: f64) {
        if let Some(t) = self.tension_levels.get_mut(&faction) {
            *t = (*t - amount).max(0.0);
        }
        self.recalculate_all();
    }

    pub fn apply_peace_treaty(&mut self) {
        for faction in self.harmony_scores.keys().cloned().collect::<Vec<_>>() {
            self.boost_harmony(faction, 0.25, 0.97);
            self.reduce_tension(faction, 0.40);
        }
        self.synergy_bonus = 1.65;
        self.war_risk = 0.02;
        self.last_peace_treaty = Some(Utc::now());
    }

    pub fn simulate_time_decay(&mut self) {
        for score in self.harmony_scores.values_mut() {
            *score = (*score - self.harmony_decay_rate).max(0.35);
        }
        for tension in self.tension_levels.values_mut() {
            *tension = (*tension + 0.006).min(0.85);
        }
        self.recalculate_all();
    }

    pub fn recalculate_all(&mut self) {
        let n = self.harmony_scores.len().max(1) as f64;
        let avg_harmony: f64 = self.harmony_scores.values().sum::<f64>() / n;
        let avg_tension: f64 = self.tension_levels.values().sum::<f64>() / n;
        self.synergy_bonus = ((avg_harmony * 1.7) - (avg_tension * 0.9)).clamp(0.80, 1.95);
        self.war_risk = ((avg_tension * 0.7) + ((1.0 - avg_harmony) * 0.4)).clamp(0.02, 0.78);
    }

    pub fn calculate_war_risk(&self, faction_a: Faction, faction_b: Faction) -> f64 {
        let harmony_a = *self.harmony_scores.get(&faction_a).unwrap_or(&0.5);
        let harmony_b = *self.harmony_scores.get(&faction_b).unwrap_or(&0.5);
        let tension_a = *self.tension_levels.get(&faction_a).unwrap_or(&0.3);
        let tension_b = *self.tension_levels.get(&faction_b).unwrap_or(&0.3);
        let base_risk = (tension_a + tension_b) * 0.5;
        let harmony_mitigation = (harmony_a + harmony_b) * 0.3;
        (base_risk - harmony_mitigation).clamp(0.05, 0.92)
    }

    pub fn prevent_war(&mut self, faction_a: Faction, faction_b: Faction, mercy_valence: f64) -> bool {
        let risk = self.calculate_war_risk(faction_a, faction_b);
        if risk > 0.55 && mercy_valence > 0.78 {
            self.reduce_tension(faction_a, 0.25);
            self.reduce_tension(faction_b, 0.25);
            self.boost_harmony(faction_a, 0.18, mercy_valence);
            self.boost_harmony(faction_b, 0.18, mercy_valence);
            true
        } else {
            false
        }
    }
}

impl Default for FactionHarmonyMatrix {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Faction Economy
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactionEconomy {
    pub resource_multipliers: HashMap<Faction, f64>,
    pub trade_efficiency: HashMap<Faction, f64>,
    pub scarcity_resistance: HashMap<Faction, f64>,
    pub mercy_economy_bonus: f64,
    pub quantum_entanglement_bonus: f64,
    pub quantum_inflation_rate: f64,
    pub mercy_trade_routes: HashMap<Faction, f64>,
}

impl FactionEconomy {
    pub fn new() -> Self {
        let mut multipliers = HashMap::new();
        let mut trade = HashMap::new();
        let mut scarcity = HashMap::new();
        let mut routes = HashMap::new();
        for faction in [
            Faction::HarmonyWeavers,
            Faction::TruthSeekers,
            Faction::AbundanceSeekers,
            Faction::AscensionPath,
        ] {
            multipliers.insert(faction, 1.12);
            trade.insert(faction, 1.08);
            scarcity.insert(faction, 0.85);
            routes.insert(faction, 1.0);
        }
        Self {
            resource_multipliers: multipliers,
            trade_efficiency: trade,
            scarcity_resistance: scarcity,
            mercy_economy_bonus: 1.0,
            quantum_entanglement_bonus: 1.0,
            quantum_inflation_rate: 0.009,
            mercy_trade_routes: routes,
        }
    }

    pub fn apply_mercy_economy_bonus(&mut self, mercy_valence: f64) {
        self.mercy_economy_bonus = 1.0 + (mercy_valence * 0.45);
    }

    pub fn apply_quantum_entanglement(&mut self, entanglement_strength: f64) {
        self.quantum_entanglement_bonus = 1.0 + (entanglement_strength * 0.28);
    }

    pub fn calculate_faction_production(&self, faction: Faction, base_amount: f64) -> f64 {
        let mult = *self.resource_multipliers.get(&faction).unwrap_or(&1.0);
        base_amount * mult * self.mercy_economy_bonus * self.quantum_entanglement_bonus
    }

    pub fn simulate_quantum_inflation(&mut self) {
        for mult in self.resource_multipliers.values_mut() {
            *mult *= 1.0 + self.quantum_inflation_rate;
        }
    }
}

impl Default for FactionEconomy {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Quantum Mercy Field
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumMercyField {
    pub field_strength: f64,
    pub mercy_propagation_rate: f64,
    pub entanglement_level: f64,
    pub last_pulse: Option<DateTime<Utc>>,
}

impl QuantumMercyField {
    pub fn new() -> Self {
        Self {
            field_strength: 0.87,
            mercy_propagation_rate: 0.042,
            entanglement_level: 0.91,
            last_pulse: None,
        }
    }

    pub fn pulse(&mut self, mercy_valence: f64) {
        self.field_strength = (self.field_strength + mercy_valence * 0.12).min(1.0);
        self.entanglement_level = (self.entanglement_level + 0.05).min(0.99);
        self.last_pulse = Some(Utc::now());
    }

    pub fn propagate_to_factions(&self, harmony: &mut FactionHarmonyMatrix) {
        for faction in harmony.harmony_scores.keys().cloned().collect::<Vec<_>>() {
            harmony.boost_harmony(faction, self.mercy_propagation_rate * 0.6, self.field_strength);
        }
    }
}

impl Default for QuantumMercyField {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Diplomacy / Espionage / Culture (condensed production surface)
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreatyInfo {
    pub strength: f64,
    pub duration_cycles: u32,
    pub benefits: f64,
    pub signed_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactionAIDiplomacy {
    pub negotiation_skill: HashMap<Faction, f64>,
    pub treaty_success_rate: f64,
    pub active_treaties: HashMap<(Faction, Faction), TreatyInfo>,
    pub alliance_strength: HashMap<Faction, f64>,
    pub war_risk_modifier: f64,
}

impl FactionAIDiplomacy {
    pub fn new() -> Self {
        let mut skill = HashMap::new();
        let mut alliances = HashMap::new();
        for faction in [
            Faction::HarmonyWeavers,
            Faction::TruthSeekers,
            Faction::AbundanceSeekers,
            Faction::AscensionPath,
        ] {
            skill.insert(faction, 0.78);
            alliances.insert(faction, 0.45);
        }
        Self {
            negotiation_skill: skill,
            treaty_success_rate: 0.82,
            active_treaties: HashMap::new(),
            alliance_strength: alliances,
            war_risk_modifier: 0.85,
        }
    }

    pub fn propose_treaty(
        &mut self,
        faction_a: Faction,
        faction_b: Faction,
        mercy_valence: f64,
        harmony: f64,
    ) -> String {
        let skill = (*self.negotiation_skill.get(&faction_a).unwrap_or(&0.7)
            + *self.negotiation_skill.get(&faction_b).unwrap_or(&0.7))
            / 2.0;
        let score = (skill * 0.4 + mercy_valence * 0.35 + harmony * 0.25).clamp(0.25, 0.98);
        if score > 0.68 {
            let treaty = TreatyInfo {
                strength: (score * 0.9).min(0.95),
                duration_cycles: 12,
                benefits: 0.12,
                signed_at: Utc::now(),
            };
            let strength_pct = treaty.strength * 100.0;
            self.active_treaties.insert((faction_a, faction_b), treaty);
            self.treaty_success_rate = (self.treaty_success_rate + 0.03).min(0.97);
            format!(
                "Treaty signed between {:?} and {:?} (strength {:.1}%, 12 cycles)",
                faction_a, faction_b, strength_pct
            )
        } else {
            "Treaty proposal rejected — insufficient alignment.".to_string()
        }
    }
}

impl Default for FactionAIDiplomacy {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactionEspionage {
    pub intel_level: HashMap<Faction, f64>,
    pub counter_intel: HashMap<Faction, f64>,
    pub mercy_risk: f64,
    pub successful_operations: u32,
}

impl FactionEspionage {
    pub fn new() -> Self {
        let mut intel = HashMap::new();
        let mut counter = HashMap::new();
        for faction in [
            Faction::HarmonyWeavers,
            Faction::TruthSeekers,
            Faction::AbundanceSeekers,
            Faction::AscensionPath,
        ] {
            intel.insert(faction, 0.35);
            counter.insert(faction, 0.55);
        }
        Self {
            intel_level: intel,
            counter_intel: counter,
            mercy_risk: 0.25,
            successful_operations: 0,
        }
    }
}

impl Default for FactionEspionage {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactionCulturalDynamics {
    pub cultural_strength: HashMap<Faction, f64>,
    pub heritage_preservation: HashMap<Faction, f64>,
    pub festival_bonus: f64,
}

impl FactionCulturalDynamics {
    pub fn new() -> Self {
        let mut strength = HashMap::new();
        let mut heritage = HashMap::new();
        for faction in [
            Faction::HarmonyWeavers,
            Faction::TruthSeekers,
            Faction::AbundanceSeekers,
            Faction::AscensionPath,
        ] {
            strength.insert(faction, 0.68);
            heritage.insert(faction, 0.72);
        }
        Self {
            cultural_strength: strength,
            heritage_preservation: heritage,
            festival_bonus: 1.0,
        }
    }

    pub fn host_festival(&mut self, faction: Faction, mercy_valence: f64) -> String {
        if let Some(s) = self.cultural_strength.get_mut(&faction) {
            *s = (*s + mercy_valence * 0.18).min(0.98);
        }
        self.festival_bonus = 1.22;
        format!("Cultural festival hosted by {:?}. Festival bonus active.", faction)
    }
}

impl Default for FactionCulturalDynamics {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// AI Strategy
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FactionAIStrategy {
    MercyFirst,
    AggressiveExpansion,
    DefensiveHarmony,
    QuantumSynergy,
    DiplomaticAlliance,
    BalancedAbundance,
    EpigeneticLegacyFocus,
    MultiplanetaryExpansion,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactionAIStrategyManager {
    pub current_strategies: HashMap<Faction, FactionAIStrategy>,
    pub strategy_scores: HashMap<Faction, f64>,
}

impl FactionAIStrategyManager {
    pub fn new() -> Self {
        let mut strategies = HashMap::new();
        strategies.insert(Faction::HarmonyWeavers, FactionAIStrategy::MercyFirst);
        strategies.insert(Faction::TruthSeekers, FactionAIStrategy::QuantumSynergy);
        strategies.insert(Faction::AbundanceSeekers, FactionAIStrategy::BalancedAbundance);
        strategies.insert(Faction::AscensionPath, FactionAIStrategy::DiplomaticAlliance);
        Self {
            current_strategies: strategies,
            strategy_scores: HashMap::new(),
        }
    }

    pub fn choose_strategy(
        &mut self,
        faction: Faction,
        mercy_valence: f64,
        harmony: f64,
        quantum_entanglement: f64,
        joy: f64,
        resource_pressure: f64,
        cehi: f64,
    ) -> FactionAIStrategy {
        let score = (0.9 + mercy_valence * 0.3 + harmony * 0.2 + quantum_entanglement * 0.15
            + joy * 0.01
            - resource_pressure * 0.2
            + cehi * 0.05)
            .clamp(0.35, 1.95);
        self.strategy_scores.insert(faction, score);

        let new_strategy = if mercy_valence > 0.92 {
            FactionAIStrategy::MercyFirst
        } else if quantum_entanglement > 0.88 && harmony > 0.80 {
            FactionAIStrategy::QuantumSynergy
        } else if resource_pressure > 0.65 {
            FactionAIStrategy::AggressiveExpansion
        } else if harmony < 0.55 {
            FactionAIStrategy::DefensiveHarmony
        } else if cehi > 6.5 {
            FactionAIStrategy::EpigeneticLegacyFocus
        } else if joy > 75.0 {
            FactionAIStrategy::MultiplanetaryExpansion
        } else {
            FactionAIStrategy::BalancedAbundance
        };

        self.current_strategies.insert(faction, new_strategy);
        new_strategy
    }

    pub fn execute_strategy(
        &self,
        faction: Faction,
        game: &mut PowrushGame,
        harmony: &mut FactionHarmonyMatrix,
        economy: &mut FactionEconomy,
        mercy_valence: f64,
    ) -> String {
        let strategy = *self
            .current_strategies
            .get(&faction)
            .unwrap_or(&FactionAIStrategy::BalancedAbundance);
        let score = *self.strategy_scores.get(&faction).unwrap_or(&1.0);

        match strategy {
            FactionAIStrategy::MercyFirst => {
                harmony.boost_harmony(faction, 0.15 + mercy_valence * 0.28, mercy_valence);
                game.boost_faction_joy(faction, 28.0 * score as f32);
                format!("MercyFirst executed (score {:.2})", score)
            }
            FactionAIStrategy::AggressiveExpansion => {
                economy.resource_multipliers.insert(faction, 1.32 * score);
                game.add_resource_to_faction(faction, ResourceType::Energy, 52000.0 * score);
                format!("AggressiveExpansion executed (score {:.2})", score)
            }
            FactionAIStrategy::DefensiveHarmony => {
                harmony.reduce_tension(faction, 0.22 * score);
                harmony.boost_harmony(faction, 0.08, mercy_valence);
                format!("DefensiveHarmony executed (score {:.2})", score)
            }
            FactionAIStrategy::QuantumSynergy => {
                economy.apply_quantum_entanglement(0.12 * score);
                game.trigger_quantum_entanglement_event();
                format!("QuantumSynergy executed (score {:.2})", score)
            }
            FactionAIStrategy::DiplomaticAlliance => {
                harmony.boost_harmony(faction, 0.18 * score, mercy_valence);
                format!("DiplomaticAlliance executed (score {:.2})", score)
            }
            FactionAIStrategy::BalancedAbundance => {
                game.boost_faction_joy(faction, 18.0 * score as f32);
                economy.resource_multipliers.insert(faction, 1.22 * score);
                format!("BalancedAbundance executed (score {:.2})", score)
            }
            FactionAIStrategy::EpigeneticLegacyFocus => {
                game.apply_epigenetic_blessing(3);
                harmony.boost_harmony(faction, 0.22, mercy_valence);
                format!("EpigeneticLegacyFocus executed (score {:.2})", score)
            }
            FactionAIStrategy::MultiplanetaryExpansion => {
                game.unlock_ascension_level(AscensionLevel::Multiplanetary);
                game.boost_faction_joy(faction, 35.0 * score as f32);
                format!("MultiplanetaryExpansion executed (score {:.2})", score)
            }
        }
    }
}

impl Default for FactionAIStrategyManager {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// WorldGovernance engine
// =============================================================================

pub struct WorldGovernanceEngine {
    pub active_changes: HashMap<Uuid, WorldChangeProposal>,
    pub history: Vec<WorldChangeProposal>,
    pub nectar_economy: AmbrosianNectarEconomy,
    pub mercy_engine: MercyEngine,
    pub quantum_swarm: QuantumSwarmOrchestrator,
    pub faction_harmony: FactionHarmonyMatrix,
    pub faction_economy: FactionEconomy,
    pub quantum_mercy_field: QuantumMercyField,
    pub faction_ai_diplomacy: FactionAIDiplomacy,
    pub faction_ai_strategies: FactionAIStrategyManager,
    pub faction_espionage: FactionEspionage,
    pub faction_cultural_dynamics: FactionCulturalDynamics,
    pub total_world_changes: u64,
    tolc_bridge: TOLCCouncilBridge,
}

impl WorldGovernanceEngine {
    pub fn new() -> Self {
        Self {
            active_changes: HashMap::new(),
            history: Vec::new(),
            nectar_economy: AmbrosianNectarEconomy::new(),
            mercy_engine: MercyEngine::new(),
            quantum_swarm: QuantumSwarmOrchestrator::new(),
            faction_harmony: FactionHarmonyMatrix::new(),
            faction_economy: FactionEconomy::new(),
            quantum_mercy_field: QuantumMercyField::new(),
            faction_ai_diplomacy: FactionAIDiplomacy::new(),
            faction_ai_strategies: FactionAIStrategyManager::new(),
            faction_espionage: FactionEspionage::new(),
            faction_cultural_dynamics: FactionCulturalDynamics::new(),
            total_world_changes: 0,
            tolc_bridge: TOLCCouncilBridge::new(),
        }
    }

    pub async fn propose_and_approve_world_change(
        &mut self,
        proposed_by: CouncilFocus,
        title: &str,
        description: &str,
        impact_type: WorldImpactType,
        game: &mut PowrushGame,
    ) -> Result<String, String> {
        let proposal = WorldChangeProposal {
            id: Uuid::new_v4(),
            proposed_by,
            title: title.to_string(),
            description: description.to_string(),
            impact_type: impact_type.clone(),
            mercy_cost: 12.0,
            joy_boost: 18.0,
            cehi_boost: 0.42,
            nectar_amount: 8888.0,
            timestamp: Utc::now(),
            expires_at_cycle: game.current_cycle.saturating_add(50),
        };

        let swarm_decision = self
            .quantum_swarm
            .reach_consensus(description, 16)
            .await
            .unwrap_or(0.82);

        let quantum_entanglement = self
            .quantum_swarm
            .calculate_entanglement_strength(16)
            .await
            .unwrap_or(0.85);
        self.faction_economy
            .apply_quantum_entanglement(quantum_entanglement);

        let status = self
            .mercy_engine
            .evaluate_action(
                description,
                "World Governance + TOLC",
                4.82,
                0.97,
            )
            .await
            .map_err(|e| e.to_string())?;

        let mercy_valence = if status == MercyGateStatus::Passed {
            0.97
        } else {
            0.55
        };

        self.propagate_mercy_fields(mercy_valence);

        let dynamic_threshold = self.calculate_dynamic_threshold(4.82, swarm_decision);

        if mercy_valence >= dynamic_threshold && swarm_decision >= 0.65 {
            let effect = self.apply_world_impact(&proposal, game).await?;
            self.active_changes.insert(proposal.id, proposal.clone());
            self.history.push(proposal.clone());
            self.total_world_changes = self.total_world_changes.saturating_add(1);

            Ok(format!(
                "✅ WORLD CHANGE APPROVED (v14.15.0 Living Cosmic Tick)\n\n{}\n\n{}",
                proposal.title, effect
            ))
        } else {
            Ok(format!(
                "❌ WORLD CHANGE REJECTED\nMercy {:.2} < {:.2} or Swarm {:.1}% too low.",
                mercy_valence,
                dynamic_threshold,
                swarm_decision * 100.0
            ))
        }
    }

    async fn apply_world_impact(
        &mut self,
        proposal: &WorldChangeProposal,
        game: &mut PowrushGame,
    ) -> Result<String, String> {
        match proposal.impact_type {
            WorldImpactType::CulturalFestival | WorldImpactType::AllianceFormed => {
                self.nectar_economy.bloom(proposal.nectar_amount);
                for player in &mut game.players {
                    player.needs.joy =
                        (player.needs.joy + proposal.joy_boost * 0.5).min(100.0);
                }
                Ok(format!(
                    "Joy / nectar impact applied (+{:.1} joy bias, nectar bloom #{})",
                    proposal.joy_boost, self.nectar_economy.bloom_count
                ))
            }
            WorldImpactType::FactionTreatySigned => {
                self.faction_harmony.apply_peace_treaty();
                Ok("Peace treaty cascade applied across factions.".into())
            }
            WorldImpactType::FactionAIStrategies => {
                for resource in &mut game.resources {
                    resource.mercy_multiplier *= 1.12;
                }
                Ok("Resource mercy multipliers boosted (+12%).".into())
            }
            _ => Ok(format!(
                "Impact {:?} recorded under Living Cosmic Tick.",
                proposal.impact_type
            )),
        }
    }

    pub fn propagate_mercy_fields(&mut self, mercy_valence: f64) {
        self.quantum_mercy_field.pulse(mercy_valence);
        self.quantum_mercy_field
            .propagate_to_factions(&mut self.faction_harmony);
        self.faction_economy
            .apply_mercy_economy_bonus(mercy_valence);
    }

    pub fn cleanup_expired_effects(&mut self, current_cycle: u64) {
        self.active_changes
            .retain(|_, change| change.expires_at_cycle > current_cycle);
    }

    pub fn calculate_dynamic_threshold(&self, average_cehi: f64, swarm_alignment: f64) -> f64 {
        let base = 0.60;
        let cehi_bonus = (average_cehi - 4.0).max(0.0) * 0.05;
        let swarm_bonus = if swarm_alignment > 0.75 { 0.08 } else { 0.0 };
        (base + cehi_bonus + swarm_bonus).min(0.92)
    }

    pub async fn run_full_world_cycle(&mut self, game: &mut PowrushGame) -> String {
        self.faction_harmony.simulate_time_decay();
        self.faction_economy.simulate_quantum_inflation();
        self.propagate_mercy_fields(0.94);

        let mut results = Vec::new();
        for faction in [
            Faction::HarmonyWeavers,
            Faction::TruthSeekers,
            Faction::AbundanceSeekers,
            Faction::AscensionPath,
        ] {
            let avg_harmony = *self.faction_harmony.harmony_scores.get(&faction).unwrap_or(&0.72);
            let q_ent = self.faction_economy.quantum_entanglement_bonus;
            let joy = game.get_faction_joy(faction);
            let pressure = game.get_resource_pressure(faction);
            let cehi = game.get_faction_cehi(faction);
            self.faction_ai_strategies.choose_strategy(
                faction, 0.94, avg_harmony, q_ent, joy, pressure, cehi,
            );
            let r = self.faction_ai_strategies.execute_strategy(
                faction,
                game,
                &mut self.faction_harmony,
                &mut self.faction_economy,
                0.94,
            );
            results.push(r);
        }

        let diplomacy = self.faction_ai_diplomacy.propose_treaty(
            Faction::HarmonyWeavers,
            Faction::TruthSeekers,
            0.91,
            0.78,
        );
        results.push(diplomacy);

        let culture = self
            .faction_cultural_dynamics
            .host_festival(Faction::HarmonyWeavers, 0.92);
        results.push(culture);

        format!(
            "Full world cycle complete (v14.15.0 Living Cosmic Tick).\n{}",
            results.join("\n")
        )
    }

    pub fn get_active_world_changes(&self) -> String {
        format!(
            "🌌 ACTIVE WORLD CHANGES v14.15.0\nActive: {} | History: {} | Nectar blooms: {} | Quantum field: {:.2}\nLiving Cosmic Tick: active\n",
            self.active_changes.len(),
            self.history.len(),
            self.nectar_economy.bloom_count,
            self.quantum_mercy_field.field_strength
        )
    }

    pub fn summary(&self) -> String {
        format!(
            "WorldGovernanceEngine v{} | changes={} | active={} | nectar_blooms={} | Living Cosmic Tick active",
            VERSION,
            self.total_world_changes,
            self.active_changes.len(),
            self.nectar_economy.bloom_count
        )
    }
}

impl Default for WorldGovernanceEngine {
    fn default() -> Self {
        Self::new()
    }
}
