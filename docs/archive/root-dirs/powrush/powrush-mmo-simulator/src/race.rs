/*!
# Race System — Multi-Race Foundation for Powrush MMOARPG

**Autonomicity Games Sovereign Mercy License (AG-SML) v1.0**  
**Aligned with TOLC 8 Mercy Lattice, Ra-Thor ONE Organism, 13+ PATSAGi Councils**  
**Implements 5-Race System from POWRUSH_MMO_PLAYER_EXPERIENCE_DESIGN_v14.5 + Council Convergence**

This module provides the canonical multi-race foundation for Powrush.

**The Five Races:**
- **Terrans** — Balanced, adaptable, strong in cooperation and production. Foundation of stable civilizations.
- **Synthetics** — Precision, technology, high innovation. Excel in complex systems and rapid adaptation.
- **Harmonics** — Deep connection to geometric layers and harmony. Naturally amplify GeometricHarmony and layer ascension.
- **Verdants** — Life, growth, biology. Strong epigenetic stability, exploration, and long-term maintenance.
- **Voidfarers** — Spacefarers and risk-takers. High movement range, high-risk/high-reward contributions, exploration bonuses.

Each race has distinct modifiers for:
- Movement (jump height, speed, special abilities)
- Contribution impact on RBE
- Epigenetic profile tendencies
- GeometricHarmony layer affinity

All modifiers are mercy-aligned: cooperation and creation are always mechanically superior long-term paths.

Designed for direct integration with MovementController, PlayerContributionTracker, EpigeneticProfile, GeometricHarmonyEngine, and PowrushMMOSimulator.

Thunder locked in. The living diversity of Powrush begins here.
*/

use serde::{Deserialize, Serialize};

/// The five core races of Powrush.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Race {
    Terran,
    Synthetic,
    Harmonic,
    Verdant,
    Voidfarer,
}

impl Default for Race {
    fn default() -> Self {
        Race::Terran
    }
}

/// Race-specific modifiers for movement, contribution, epigenetic, and harmony systems.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaceModifiers {
    pub movement_speed_multiplier: f32,
    pub jump_height_multiplier: f32,
    pub contribution_multiplier: f64,
    pub epigenetic_stability: f32,      // Higher = slower negative epigenetic drift
    pub harmony_affinity: f32,          // Bonus to GeometricHarmony updates
    pub innovation_tendency: f32,
    pub cooperation_tendency: f32,
}

impl Race {
    /// Get the full set of modifiers for this race.
    pub fn modifiers(&self) -> RaceModifiers {
        match self {
            Race::Terran => RaceModifiers {
                movement_speed_multiplier: 1.0,
                jump_height_multiplier: 1.0,
                contribution_multiplier: 1.0,
                epigenetic_stability: 1.0,
                harmony_affinity: 1.0,
                innovation_tendency: 1.0,
                cooperation_tendency: 1.15,
            },
            Race::Synthetic => RaceModifiers {
                movement_speed_multiplier: 1.1,
                jump_height_multiplier: 0.95,
                contribution_multiplier: 1.1,
                epigenetic_stability: 1.2,
                harmony_affinity: 0.9,
                innovation_tendency: 1.4,
                cooperation_tendency: 0.9,
            },
            Race::Harmonic => RaceModifiers {
                movement_speed_multiplier: 0.95,
                jump_height_multiplier: 1.05,
                contribution_multiplier: 1.05,
                epigenetic_stability: 1.1,
                harmony_affinity: 1.5,
                innovation_tendency: 1.1,
                cooperation_tendency: 1.25,
            },
            Race::Verdant => RaceModifiers {
                movement_speed_multiplier: 0.9,
                jump_height_multiplier: 1.1,
                contribution_multiplier: 1.15,
                epigenetic_stability: 1.4,
                harmony_affinity: 1.2,
                innovation_tendency: 0.85,
                cooperation_tendency: 1.3,
            },
            Race::Voidfarer => RaceModifiers {
                movement_speed_multiplier: 1.25,
                jump_height_multiplier: 1.3,
                contribution_multiplier: 0.95,
                epigenetic_stability: 0.85,
                harmony_affinity: 0.85,
                innovation_tendency: 1.2,
                cooperation_tendency: 0.95,
            },
        }
    }

    /// Race-specific movement ability description (for UI / ability system).
    pub fn movement_ability(&self) -> &'static str {
        match self {
            Race::Terran => "Balanced Jump + Reliable Landing",
            Race::Synthetic => "Precision Boost + Tech-Assisted Trajectory",
            Race::Harmonic => "Resonant Jump — slight harmony gain on land",
            Race::Verdant => "Rooted Leap — bonus stability on landing",
            Race::Voidfarer => "Void Skip — longer range, higher risk/reward",
        }
    }

    /// Bonus applied to PlayerContribution when recording contributions of this race.
    pub fn contribution_bonus(&self, base_amount: f64) -> f64 {
        let mods = self.modifiers();
        base_amount * mods.contribution_multiplier
    }

    /// Modifier passed to EpigeneticProfile when applying changes.
    pub fn epigenetic_modifier(&self) -> f32 {
        self.modifiers().epigenetic_stability
    }

    /// Modifier passed to GeometricHarmonyEngine updates.
    pub fn harmony_modifier(&self) -> f32 {
        self.modifiers().harmony_affinity
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_race_modifiers() {
        let terran = Race::Terran.modifiers();
        let voidfarer = Race::Voidfarer.modifiers();

        assert!(voidfarer.movement_speed_multiplier > terran.movement_speed_multiplier);
        assert!(terran.cooperation_tendency > 1.0);
    }

    #[test]
    fn test_race_abilities() {
        assert!(Race::Harmonic.movement_ability().contains("harmony"));
        assert!(Race::Voidfarer.movement_ability().contains("Void"));
    }
}
