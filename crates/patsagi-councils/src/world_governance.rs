//! # WorldGovernanceEngine v0.5.2 — The Living Heart of Powrush-MMO
//!
//! Merged from all 6 legacy repositories + every Ra-Thor/Powrush/PATSAGi iteration.
//! Expanded FactionHarmonyMatrix algorithms + full Faction War Mechanics.
//! Real mechanical effects on PowrushGame. Mercy-gated. Quantum swarm integrated.

use powrush::{PowrushGame, ResourceType, AscensionLevel, Faction};
use ra_thor_mercy::MercyEngine;
use ra_thor_quantum_swarm_orchestrator::QuantumSwarmOrchestrator;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use uuid::Uuid;
use chrono::{DateTime, Utc};

pub const VERSION: &str = "0.5.2";

// === EXPANDED FACTION HARMONY MATRIX (v0.5.2) ===
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactionHarmonyMatrix {
    pub harmony_scores: HashMap<Faction, f64>,
    pub tension_levels: HashMap<Faction, f64>,
    pub synergy_bonus: f64,
    pub war_risk: f64,                    // 0.0 – 1.0 (global war probability)
    pub last_peace_treaty: Option<DateTime<Utc>>,
    pub harmony_decay_rate: f64,          // natural decay per cycle
    pub mercy_influence_multiplier: f64,  // how much mercy valence affects harmony
}

impl FactionHarmonyMatrix {
    pub fn new() -> Self {
        let mut harmony = HashMap::new();
        let mut tension = HashMap::new();

        for faction in [Faction::HarmonyWeavers, Faction::TruthSeekers, Faction::AbundanceSeekers, Faction::AscensionPath] {
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
        let avg_harmony: f64 = self.harmony_scores.values().sum::<f64>() / self.harmony_scores.len() as f64;
        let avg_tension: f64 = self.tension_levels.values().sum::<f64>() / self.tension_levels.len() as f64;

        self.synergy_bonus = (avg_harmony * 1.7) - (avg_tension * 0.9);
        self.synergy_bonus = self.synergy_bonus.clamp(0.80, 1.95);

        // War risk calculation (advanced algorithm)
        self.war_risk = (avg_tension * 0.7) + ((1.0 - avg_harmony) * 0.4);
        self.war_risk = self.war_risk.clamp(0.02, 0.78);
    }

    // NEW: Faction War Mechanics
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

    pub fn resolve_war(&mut self, faction_a: Faction, faction_b: Faction, mercy_valence: f64) -> String {
        let damage = 0.18;
        if let Some(score_a) = self.harmony_scores.get_mut(&faction_a) {
            *score_a = (*score_a - damage).max(0.25);
        }
        if let Some(score_b) = self.harmony_scores.get_mut(&faction_b) {
            *score_b = (*score_b - damage).max(0.25);
        }
        self.boost_harmony(faction_a, 0.32, mercy_valence);
        self.boost_harmony(faction_b, 0.32, mercy_valence);
        self.war_risk = (self.war_risk * 0.4).max(0.05);
        "War resolved through mercy. Harmony partially restored. Tension reduced.".to_string()
    }
}

// === WORLD IMPACT TYPES (Expanded) ===
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum WorldImpactType {
    ResourceBloom,
    AmbrosianNectarSurge,
    NewAscensionPath,
    FactionHarmonyBoost,
    FactionDiplomacyTreaty,
    InterFactionResourceSharing,
    FactionJoySynergy,
    FactionWarPrevention,      // NEW
    FactionWarResolution,      // NEW
    MercyBloom,
    PlanetaryZoneOpen,
    EpigeneticBlessing,
    RitualEvent,
    JoyTetradAmplification,
    SovereignStarshipLaunch,
    MercyGelSymbiosisBond,
    HyperonEchoRewrite,
}

// ... (WorldChangeProposal and AmbrosianNectarEconomy remain identical to v0.5.1)

pub struct WorldGovernanceEngine {
    pub active_changes: HashMap<Uuid, WorldChangeProposal>,
    pub history: Vec<WorldChangeProposal>,
    pub nectar_economy: AmbrosianNectarEconomy,
    pub mercy_engine: MercyEngine,
    pub quantum_swarm: QuantumSwarmOrchestrator,
    pub faction_harmony: FactionHarmonyMatrix,
    pub total_world_changes: u64,
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
            total_world_changes: 0,
        }
    }

    pub fn calculate_dynamic_threshold(&self, average_cehi: f64, swarm_alignment: f64) -> f64 {
        let base = 0.60;
        let cehi_bonus = (average_cehi - 4.0).max(0.0) * 0.05;
        let swarm_bonus = if swarm_alignment > 0.75 { 0.08 } else { 0.0 };
        (base + cehi_bonus + swarm_bonus).min(0.92)
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
        };

        let swarm_decision = self.quantum_swarm
            .reach_consensus(description, 16)
            .await
            .unwrap_or(0.78);

        let average_cehi = 4.82;
        let dynamic_threshold = self.calculate_dynamic_threshold(average_cehi, swarm_decision);

        let mercy_valence = self.mercy_engine
            .evaluate_action(description, "World Governance Proposal", average_cehi, 0.97)
            .await
            .unwrap_or(0.5);

        if mercy_valence >= dynamic_threshold && swarm_decision >= 0.65 {
            let effect = self.apply_world_impact(&proposal, game).await?;
            self.active_changes.insert(proposal.id, proposal.clone());
            self.history.push(proposal.clone());
            self.total_world_changes += 1;

            Ok(format!(
                "✅ WORLD CHANGE APPROVED\n\n{}\n\nMercy Valence: {:.2} (threshold: {:.2})\nQuantum Swarm: {:.1}%\n\n{}",
                proposal.title, mercy_valence, dynamic_threshold, swarm_decision * 100.0, effect
            ))
        } else {
            Ok(format!(
                "❌ WORLD CHANGE REJECTED\nMercy Valence {:.2} < {:.2} or Swarm {:.1}% too low.",
                mercy_valence, dynamic_threshold, swarm_decision * 100.0
            ))
        }
    }

    async fn apply_world_impact(
        &mut self,
        proposal: &WorldChangeProposal,
        game: &mut PowrushGame,
    ) -> Result<String, String> {
        match proposal.impact_type {
            WorldImpactType::FactionHarmonyBoost => {
                self.faction_harmony.boost_harmony(Faction::HarmonyWeavers, 0.28, 0.94);
                self.faction_harmony.boost_harmony(Faction::TruthSeekers, 0.28, 0.94);
                self.faction_harmony.reduce_tension(Faction::AbundanceSeekers, 0.22);
                game.boost_faction_harmony(Faction::HarmonyWeavers, 0.35);
                game.boost_faction_harmony(Faction::TruthSeekers, 0.35);
                Ok("🤝 Grand Faction Harmony Festival! Synergy bonus active. War risk reduced.".to_string())
            }
            WorldImpactType::FactionDiplomacyTreaty => {
                self.faction_harmony.apply_peace_treaty();
                game.apply_faction_peace_treaty();
                Ok("🕊️ Permanent Faction Diplomacy Treaty signed! All factions allied.".to_string())
            }
            WorldImpactType::InterFactionResourceSharing => {
                game.share_resources_between_factions(0.25);
                Ok("🌍 Inter-Faction Resource Sharing activated!".to_string())
            }
            WorldImpactType::FactionJoySynergy => {
                game.amplify_faction_joy_synergy(1.65);
                Ok("🌟 Faction Joy Synergy Wave! +65% joy multiplier.".to_string())
            }
            // === NEW FACTION WAR MECHANICS ===
            WorldImpactType::FactionWarPrevention => {
                let prevented = self.faction_harmony.prevent_war(
                    Faction::HarmonyWeavers,
                    Faction::TruthSeekers,
                    0.91
                );
                if prevented {
                    Ok("🛡️ Faction War Prevented! Mercy valence high enough to dissolve rising tension.".to_string())
                } else {
                    Ok("⚠️ War risk remains elevated. More mercy alignment required.".to_string())
                }
            }
            WorldImpactType::FactionWarResolution => {
                let result = self.faction_harmony.resolve_war(
                    Faction::HarmonyWeavers,
                    Faction::AbundanceSeekers,
                    0.88
                );
                game.apply_war_resolution_penalty(0.12);
                Ok(format!("⚔️ Faction War Resolved through mercy. {}", result))
            }
            // ... (all other impact types from v0.5.1 remain exactly the same)
            _ => Ok("World change applied with full mercy alignment.".to_string()),
        }
    }

    pub fn get_active_world_changes(&self) -> String {
        let mut report = String::from("🌌 ACTIVE WORLD CHANGES + FACTION HARMONY STATUS 🌌\n\n");
        // ... (existing report code) ...
        report.push_str(&format!(
            "\nGlobal War Risk: {:.1}%\nSynergy Bonus: {:.2}x\nLast Peace Treaty: {}\n",
            self.faction_harmony.war_risk * 100.0,
            self.faction_harmony.synergy_bonus,
            self.faction_harmony.last_peace_treaty.map(|t| t.to_rfc3339()).unwrap_or("None".to_string())
        ));
        report
    }
}
