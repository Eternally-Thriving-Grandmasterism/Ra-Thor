//! Governance Types — v14.15.0
//!
//! Compatibility surface so PATSAGi Councils / WorldGovernance can drive
//! the Powrush organism without depending on unfinished MMO modules.
//!
//! Living Cosmic Tick aligned. Contact: info@Rathor.ai

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Faction {
    HarmonyWeavers,
    TruthSeekers,
    AbundanceSeekers,
    AscensionPath,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MercyGateStatus {
    Passed,
    Rejected,
    Deferred,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ResourceType {
    Energy,
    Knowledge,
    Matter,
    Joy,
    Mercy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AscensionLevel {
    Planetary,
    Multiplanetary,
    Interstellar,
    Cosmic,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlayerNeeds {
    pub joy: f32,
    pub harmony: f32,
    pub abundance: f32,
}

impl Default for PlayerNeeds {
    fn default() -> Self {
        Self {
            joy: 75.0,
            harmony: 70.0,
            abundance: 65.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Player {
    pub id: u64,
    pub name: String,
    pub happiness: f32,
    pub needs: PlayerNeeds,
    pub faction: Faction,
    pub cehi: f64,
}

impl Player {
    pub fn new(id: u64, name: impl Into<String>, faction: Faction) -> Self {
        Self {
            id,
            name: name.into(),
            happiness: 75.0,
            needs: PlayerNeeds::default(),
            faction,
            cehi: 4.5,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldResource {
    pub kind: ResourceType,
    pub amount: f64,
    pub mercy_multiplier: f64,
}

impl WorldResource {
    pub fn new(kind: ResourceType, amount: f64) -> Self {
        Self {
            kind,
            amount,
            mercy_multiplier: 1.0,
        }
    }
}

/// Authoritative-enough game state for PATSAGi governance ticks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowrushGame {
    pub current_cycle: u64,
    pub players: Vec<Player>,
    pub resources: Vec<WorldResource>,
    pub faction_joy: std::collections::HashMap<Faction, f64>,
    pub faction_pressure: std::collections::HashMap<Faction, f64>,
    pub faction_cehi: std::collections::HashMap<Faction, f64>,
    pub ascension: AscensionLevel,
    pub quantum_entanglement_events: u64,
    pub epigenetic_blessings: u32,
}

impl PowrushGame {
    pub fn new() -> Self {
        let mut faction_joy = std::collections::HashMap::new();
        let mut faction_pressure = std::collections::HashMap::new();
        let mut faction_cehi = std::collections::HashMap::new();
        for f in [
            Faction::HarmonyWeavers,
            Faction::TruthSeekers,
            Faction::AbundanceSeekers,
            Faction::AscensionPath,
        ] {
            faction_joy.insert(f, 72.0);
            faction_pressure.insert(f, 0.25);
            faction_cehi.insert(f, 4.5);
        }

        Self {
            current_cycle: 0,
            players: vec![
                Player::new(1, "Seeker-Alpha", Faction::HarmonyWeavers),
                Player::new(2, "Seeker-Beta", Faction::TruthSeekers),
            ],
            resources: vec![
                WorldResource::new(ResourceType::Energy, 100_000.0),
                WorldResource::new(ResourceType::Knowledge, 50_000.0),
                WorldResource::new(ResourceType::Joy, 25_000.0),
            ],
            faction_joy,
            faction_pressure,
            faction_cehi,
            ascension: AscensionLevel::Planetary,
            quantum_entanglement_events: 0,
            epigenetic_blessings: 0,
        }
    }

    pub async fn run_simulation_cycle(&mut self) -> Result<String, String> {
        self.current_cycle = self.current_cycle.saturating_add(1);
        Ok(format!(
            "PowrushGame cycle {} complete | players={} | Living Cosmic Tick",
            self.current_cycle,
            self.players.len()
        ))
    }

    pub fn get_faction_joy(&self, faction: Faction) -> f64 {
        *self.faction_joy.get(&faction).unwrap_or(&70.0)
    }

    pub fn get_resource_pressure(&self, faction: Faction) -> f64 {
        *self.faction_pressure.get(&faction).unwrap_or(&0.3)
    }

    pub fn get_faction_cehi(&self, faction: Faction) -> f64 {
        *self.faction_cehi.get(&faction).unwrap_or(&4.5)
    }

    pub fn boost_faction_joy(&mut self, faction: Faction, amount: f32) {
        let entry = self.faction_joy.entry(faction).or_insert(70.0);
        *entry = (*entry + amount as f64).min(100.0);
        for p in self.players.iter_mut().filter(|p| p.faction == faction) {
            p.happiness = (p.happiness + amount).min(100.0);
            p.needs.joy = (p.needs.joy + amount * 0.5).min(100.0);
        }
    }

    pub fn add_resource_to_faction(&mut self, _faction: Faction, kind: ResourceType, amount: f64) {
        if let Some(r) = self.resources.iter_mut().find(|r| r.kind == kind) {
            r.amount += amount;
        } else {
            self.resources.push(WorldResource::new(kind, amount));
        }
    }

    pub fn trigger_quantum_entanglement_event(&mut self) {
        self.quantum_entanglement_events = self.quantum_entanglement_events.saturating_add(1);
    }

    pub fn apply_epigenetic_blessing(&mut self, generations: u32) {
        self.epigenetic_blessings = self.epigenetic_blessings.saturating_add(generations);
    }

    pub fn unlock_ascension_level(&mut self, level: AscensionLevel) {
        self.ascension = level;
    }
}

impl Default for PowrushGame {
    fn default() -> Self {
        Self::new()
    }
}
