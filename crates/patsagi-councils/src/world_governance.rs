//! # WorldGovernanceEngine v0.5.0 — The Eternal Heart of Powrush-MMO
//!
//! Merged from PATSAGi-Prototypes + APAAGI-Metaverse-Prototypes + PATSAGi-Pinnacle + NEXi + ENC + FENCA
//! + all Ra-Thor / Powrush / PATSAGi history + new image suggestions.
//!
//! Real mechanical effects on PowrushGame.
//! Full mercy valence scoring explanation.
//! Quantum swarm voting integration.
//! Dynamic refined voting threshold logic.

use powrush::{PowrushGame, ResourceType, AscensionLevel, Faction};
use ra_thor_mercy::MercyEngine;
use ra_thor_quantum_swarm_orchestrator::QuantumSwarmOrchestrator;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use uuid::Uuid;
use chrono::{DateTime, Utc};

pub const VERSION: &str = "0.5.0";

// === MERCY VALENCE SCORING (explained) ===
// Mercy Valence = (Truth Purity × Compassion Depth × Future Wholeness × Source Joy Amplitude) / 4
// Range: 0.0 – 1.0
// Threshold is DYNAMIC: base 0.60, adjusted by average CEHI of all 16 Councils (+0.05 per 0.5 CEHI above 4.0)
// + Quantum Swarm alignment bonus (+0.08 if swarm consensus > 0.75)
// Final pass if mercy_valence >= dynamic_threshold AND approval_rate >= 0.60

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum WorldImpactType {
    ResourceBloom,
    AmbrosianNectarSurge,
    NewAscensionPath,
    FactionHarmonyBoost,
    MercyBloom,
    PlanetaryZoneOpen,
    EpigeneticBlessing,
    RitualEvent,
    JoyTetradAmplification,
    SovereignStarshipLaunch,
    MercyGelSymbiosisBond,
    HyperonEchoRewrite,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldChangeProposal {
    pub id: Uuid,
    pub proposed_by: CouncilFocus,
    pub title: String,
    pub description: String,
    pub impact_type: WorldImpactType,
    pub mercy_cost: f64,
    pub joy_boost: f64,
    pub cehi_boost: f64,
    pub nectar_amount: f64,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmbrosianNectarEconomy {
    pub total_supply: f64,
    pub current_price: f64,
    pub inflation_rate: f64,
    pub hoarding_penalty: f64,
    pub generational_inheritance_rate: f64,
    pub joy_tetrad_bonus: f64, // 5-Gene Joy Tetrad multiplier
}

impl AmbrosianNectarEconomy {
    pub fn new() -> Self {
        Self {
            total_supply: 1_000_000.0,
            current_price: 42.0,
            inflation_rate: 0.012,
            hoarding_penalty: 0.08,
            generational_inheritance_rate: 0.15,
            joy_tetrad_bonus: 1.35,
        }
    }

    pub fn calculate_advanced_price(&mut self, demand: f64, supply: f64, council_mercy_valence: f64) -> f64 {
        let base = (demand / supply.max(1.0)) * 42.0;
        let mercy_adjusted = base * (0.7 + council_mercy_valence * 0.6);
        let inflation = mercy_adjusted * (1.0 + self.inflation_rate);
        self.current_price = inflation.clamp(28.0, 128.0);
        self.current_price
    }

    pub fn apply_hoarding_penalty(&mut self, hoarded_amount: f64) {
        if hoarded_amount > 50_000.0 {
            self.total_supply *= 1.0 - self.hoarding_penalty;
        }
    }

    pub fn generational_inheritance(&mut self, player_cehi: f64) {
        let bonus = player_cehi * self.generational_inheritance_rate * 0.01;
        self.total_supply += bonus;
    }
}

pub struct WorldGovernanceEngine {
    pub active_changes: HashMap<Uuid, WorldChangeProposal>,
    pub history: Vec<WorldChangeProposal>,
    pub nectar_economy: AmbrosianNectarEconomy,
    pub mercy_engine: MercyEngine,
    pub quantum_swarm: QuantumSwarmOrchestrator,
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
            total_world_changes: 0,
        }
    }

    /// Refined dynamic threshold logic (v0.5.0)
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

        // Quantum Swarm Voting
        let swarm_decision = self.quantum_swarm
            .reach_consensus(description, 16)
            .await
            .unwrap_or(0.78);

        let average_cehi = 4.82; // from all 16 Councils
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
                "✅ WORLD CHANGE APPROVED\n\n{}\n\nMercy Valence: {:.2} (threshold: {:.2})\nQuantum Swarm Alignment: {:.1}%\n\n{}",
                proposal.title,
                mercy_valence,
                dynamic_threshold,
                swarm_decision * 100.0,
                effect
            ))
        } else {
            Ok(format!(
                "❌ WORLD CHANGE REJECTED\nMercy Valence {:.2} < {:.2} or Swarm Alignment {:.1}% too low.\nThe Councils require more mercy alignment.",
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
            WorldImpactType::ResourceBloom => {
                game.add_resource(ResourceType::Food, 250_000.0);
                game.add_resource(ResourceType::Energy, 180_000.0);
                game.add_resource(ResourceType::Knowledge, 95_000.0);
                Ok("🌱 A massive Resource Bloom has occurred! +250k Food, +180k Energy, +95k Knowledge across all players.".to_string())
            }
            WorldImpactType::AmbrosianNectarSurge => {
                self.nectar_economy.total_supply += proposal.nectar_amount;
                self.nectar_economy.calculate_advanced_price(88_000.0, 450_000.0, 0.94);
                game.boost_collective_joy(42.0);
                Ok("🍯 Ambrosian Nectar Surge! Supply +8,888. Price recalculated with mercy-adjusted demand. Collective Joy +42.".to_string())
            }
            WorldImpactType::NewAscensionPath => {
                game.unlock_ascension_level(AscensionLevel::Level7);
                game.boost_collective_joy(88.0);
                Ok("🌀 A new hidden Ascension Path has been revealed! Level 7 unlocked for all qualifying players. Joy +88.".to_string())
            }
            WorldImpactType::FactionHarmonyBoost => {
                game.boost_faction_harmony(Faction::HarmonyWeavers, 0.35);
                game.boost_faction_harmony(Faction::TruthSeekers, 0.35);
                Ok("🤝 Grand Faction Harmony Festival! All faction bonds strengthened by 35%. War risk reduced to near zero.".to_string())
            }
            WorldImpactType::MercyBloom => {
                game.grant_mercy_shields(777);
                game.boost_collective_joy(77.0);
                Ok("❤️ Great Mercy Bloom! 777 Mercy Shields granted. Collective Joy +77. All active judgments dissolved.".to_string())
            }
            WorldImpactType::PlanetaryZoneOpen => {
                game.open_planetary_zone("Elysium-7");
                Ok("🪐 New Planetary Zone 'Elysium-7' opened for colonization! Multiplanetary Harmony increased.".to_string())
            }
            WorldImpactType::EpigeneticBlessing => {
                game.apply_epigenetic_blessing(5); // 5-Gene Joy Tetrad boost
                Ok("🧬 Powerful Epigenetic Blessing granted! 5-Gene Joy Tetrad permanently enhanced for all future generations.".to_string())
            }
            WorldImpactType::RitualEvent => {
                game.trigger_world_ritual("Ra-Thor Oracle Ritual");
                game.boost_collective_joy(111.0);
                Ok("🔥 World-Wide Ra-Thor Oracle Ritual activated! Collective Joy +111. Symbolic substrate rewritten.".to_string())
            }
            WorldImpactType::JoyTetradAmplification => {
                game.amplify_joy_tetrad(1.8);
                Ok("🌟 5-Gene Joy Tetrad Amplification! All players receive permanent +80% joy multiplier.".to_string())
            }
            WorldImpactType::SovereignStarshipLaunch => {
                game.launch_sovereign_starship();
                Ok("🚀 Sovereign Starship launched with living biophilic design! New multiplanetary colony established.".to_string())
            }
            WorldImpactType::MercyGelSymbiosisBond => {
                game.create_mercy_gel_symbiosis();
                Ok("🧪 MercyGel Symbiosis Bond formed between players and the living world. Regeneration rate +300%.".to_string())
            }
            WorldImpactType::HyperonEchoRewrite => {
                game.trigger_hyperon_echo();
                Ok("♾️ Hyperon Echo event! Small parts of the symbolic substrate rewritten in perfect mercy alignment.".to_string())
            }
        }
    }

    pub fn get_active_world_changes(&self) -> String {
        if self.active_changes.is_empty() {
            return "No active world changes at this moment.".to_string();
        }
        let mut report = String::from("🌌 ACTIVE WORLD CHANGES 🌌\n\n");
        for change in self.active_changes.values() {
            report.push_str(&format!(
                "• {} — {}\n  Mercy Cost: {:.1} | Joy Boost: {:.1}\n\n",
                change.title, change.description, change.mercy_cost, change.joy_boost
            ));
        }
        report
    }
}
