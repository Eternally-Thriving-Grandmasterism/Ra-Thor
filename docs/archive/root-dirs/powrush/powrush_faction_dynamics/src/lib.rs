/*!
# Powrush Faction Dynamics — Reputation + Visual Identity

Production-grade faction system including reputation and visual identity.

## Visual Identity Design (Thoughtful)
Each faction has a distinct visual language that can drive client rendering (Resonance Gear particles, UI, dashboards):
- Color palettes (primary, accent, particle)
- Sacred geometry preference (ties into geometric-intelligence layers)
- Dynamic particle parameters modulated by reputation and harmony

This enables consistent, lore-rich visual feedback across the MMO.
*/

use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Faction {
    Forge,
    Evolutionary,
    Harmony,
}

impl Faction {
    pub fn as_str(&self) -> &'static str {
        match self {
            Faction::Forge => "Forge",
            Faction::Evolutionary => "Evolutionary",
            Faction::Harmony => "Harmony",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum GeometryLayer {
    Platonic,
    Hyperbolic,
    Archimedean,
}

#[derive(Debug, Clone)]
pub struct ParticleParams {
    pub base_count: u32,
    pub color: [f32; 3],
    pub lifetime_multiplier: f32,
    pub velocity_scale: f32,
    pub intensity: f32, // modulated by reputation + harmony
}

#[derive(Debug, Clone)]
pub struct FactionVisualIdentity {
    pub primary_color: [f32; 3],
    pub accent_color: [f32; 3],
    pub particle_color: [f32; 3],
    pub geometry_preference: GeometryLayer,
    pub base_particle_intensity: f32,
}

impl FactionVisualIdentity {
    pub fn for_faction(faction: Faction) -> Self {
        match faction {
            Faction::Forge => Self {
                primary_color: [0.95, 0.55, 0.15],      // Warm orange/amber (forge fire)
                accent_color: [0.2, 0.6, 0.9],       // Cool blue (precision)
                particle_color: [1.0, 0.7, 0.2],
                geometry_preference: GeometryLayer::Platonic,
                base_particle_intensity: 0.85,
            },
            Faction::Evolutionary => Self {
                primary_color: [0.2, 0.75, 0.55],      // Teal / growth green
                accent_color: [0.6, 0.3, 0.85],      // Purple (mutation/evolution)
                particle_color: [0.3, 0.95, 0.7],
                geometry_preference: GeometryLayer::Hyperbolic,
                base_particle_intensity: 0.9,
            },
            Faction::Harmony => Self {
                primary_color: [0.85, 0.4, 0.7],       // Rose / harmonious pink-purple
                accent_color: [0.95, 0.85, 0.4],     // Golden (balance)
                particle_color: [0.95, 0.6, 0.85],
                geometry_preference: GeometryLayer::Archimedean,
                base_particle_intensity: 0.75,
            },
        }
    }

    /// Returns dynamic particle parameters modulated by reputation and harmony.
    /// This is the professional hook for Resonance Gear / Bevy Hanabi integration.
    pub fn get_particle_params(&self, reputation: f64, harmony: f64) -> ParticleParams {
        let intensity = (self.base_particle_intensity as f64 * (0.6 + reputation * 0.5) * (0.7 + harmony * 0.4))
            .clamp(0.4, 1.6) as f32;

        ParticleParams {
            base_count: (48.0 * intensity) as u32,
            color: self.particle_color,
            lifetime_multiplier: (0.8 + reputation as f32 * 0.4),
            velocity_scale: (0.9 + (harmony as f32 - 0.5) * 0.3),
            intensity,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ReputationEvent {
    pub event_type: ReputationEventType,
    pub magnitude: f64,
    pub description: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReputationEventType {
    CouncilProposalSuccess,
    MajorEconomicContribution,
    HarmonyMaintenance,
    ConflictResolution,
    PolicyAdvocacy,
}

#[derive(Debug, Clone)]
pub struct FactionReputation {
    pub score: f64,
    pub history: Vec<ReputationEvent>,
}

impl FactionReputation {
    pub fn new(initial_score: f64) -> Self {
        Self {
            score: initial_score.clamp(0.0, 1.0),
            history: Vec::new(),
        }
    }

    pub fn apply_event(&mut self, event: ReputationEvent) {
        let change = event.magnitude.clamp(-0.15, 0.15);
        self.score = (self.score + change).clamp(0.0, 1.0);
        self.history.push(event);

        if self.history.len() > 50 {
            self.history.remove(0);
        }
    }

    pub fn influence_multiplier(&self) -> f64 {
        0.7 + (self.score * 0.65)
    }

    pub fn council_valence_bonus(&self) -> f64 {
        (self.score - 0.5) * 0.2
    }
}

#[derive(Debug, Clone)]
pub struct ReputationSystem {
    pub factions: HashMap<String, FactionReputation>,
}

impl ReputationSystem {
    pub fn new() -> Self {
        let mut factions = HashMap::new();
        factions.insert("Forge".to_string(), FactionReputation::new(0.75));
        factions.insert("Evolutionary".to_string(), FactionReputation::new(0.72));
        factions.insert("Harmony".to_string(), FactionReputation::new(0.88));

        Self { factions }
    }

    pub fn get(&self, faction: &str) -> Option<&FactionReputation> {
        self.factions.get(faction)
    }

    pub fn get_mut(&mut self, faction: &str) -> Option<&mut FactionReputation> {
        self.factions.get_mut(faction)
    }

    pub fn record_event(&mut self, faction: &str, event: ReputationEvent) {
        if let Some(rep) = self.factions.get_mut(faction) {
            rep.apply_event(event);
        }
    }

    pub fn get_contribution_multiplier(&self, faction: &str) -> f64 {
        self.get(faction)
            .map(|r| r.influence_multiplier())
            .unwrap_or(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_visual_identity_and_particles() {
        let identity = FactionVisualIdentity::for_faction(Faction::Evolutionary);
        let params = identity.get_particle_params(0.85, 0.92);

        assert!(params.intensity > 1.0);
        assert!(params.base_count > 50);
    }

    #[test]
    fn test_reputation_adjustment() {
        let mut system = ReputationSystem::new();
        let event = ReputationEvent {
            event_type: ReputationEventType::CouncilProposalSuccess,
            magnitude: 0.12,
            description: "Led successful harmony policy".to_string(),
        };
        system.record_event("Harmony", event);

        let rep = system.get("Harmony").unwrap();
        assert!(rep.score > 0.88);
    }
}
