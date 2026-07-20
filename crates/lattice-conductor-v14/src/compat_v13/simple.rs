//! SimpleLatticeConductor — v13-shaped facade over LatticeConductorV14
//!
//! Provides tick + geometric state for leaf crates while holding a real
//! v14 orchestrator. Cosmic Loop is enforced on every tick via arbitration.

use crate::compat_v13::geometric::GeometricState;
use crate::compat_v13::traits::{Conductable, ConductorRegistry};
use crate::LatticeConductorV14;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveParameters {
    pub mercy_recovery_rate: f64,
    pub evolution_rate: f64,
}

impl Default for AdaptiveParameters {
    fn default() -> Self {
        Self {
            mercy_recovery_rate: 1.0,
            evolution_rate: 0.01,
        }
    }
}

#[derive(Debug)]
pub struct SimpleLatticeConductor {
    pub name: String,
    pub state: GeometricState,
    pub adaptive_params: AdaptiveParameters,
    pub registry: ConductorRegistry,
    pub tick_count: u64,
    pub v14: LatticeConductorV14,
}

impl Clone for SimpleLatticeConductor {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            state: self.state.clone(),
            adaptive_params: self.adaptive_params.clone(),
            registry: ConductorRegistry {
                registered_systems: self.registry.registered_systems.clone(),
                tick_count: self.registry.tick_count,
            },
            tick_count: self.tick_count,
            v14: LatticeConductorV14::new(),
        }
    }
}

impl SimpleLatticeConductor {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            state: GeometricState::default(),
            adaptive_params: AdaptiveParameters::default(),
            registry: ConductorRegistry::new(),
            tick_count: 0,
            v14: LatticeConductorV14::new(),
        }
    }

    pub fn tick(&mut self) -> GeometricState {
        self.v14.enforce_cosmic_loop_activation();
        self.v14.arbitration_engine.before_council_arbitration();

        let drift = 0.008 * self.adaptive_params.mercy_recovery_rate;
        self.state.apply_mercy_drift(drift);
        self.state.evolution_level =
            (self.state.evolution_level + self.adaptive_params.evolution_rate * 0.1).min(10.0);

        self.tick_count = self.tick_count.saturating_add(1);
        self.registry.advance_tick();

        self.state.clone()
    }

    pub fn is_cosmic_loop_ready(&self) -> bool {
        self.v14.arbitration_engine.is_cosmic_loop_ready()
    }

    pub fn bless_system(&mut self, system_id: &str, mercy_alignment: f64, notes: &str) {
        self.registry
            .bless_system(system_id, mercy_alignment, notes);
    }
}

impl Default for SimpleLatticeConductor {
    fn default() -> Self {
        Self::new("simple-lattice-conductor")
    }
}

impl Conductable for SimpleLatticeConductor {
    fn system_id(&self) -> &'static str {
        "simple_lattice_conductor"
    }

    fn system_name(&self) -> &'static str {
        "SimpleLatticeConductor (v13-compat)"
    }

    fn on_conductor_tick(&mut self, conductor_state: &GeometricState) {
        self.state.mercy_score =
            (self.state.mercy_score * 0.8 + conductor_state.mercy_score * 0.2).clamp(0.3, 1.5);
        let _ = self.tick();
    }

    fn get_mercy_state(&self) -> Option<f64> {
        Some(self.state.mercy_score)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tick_enforces_cosmic_loop() {
        let mut c = SimpleLatticeConductor::new("test");
        let s = c.tick();
        assert!(c.is_cosmic_loop_ready());
        assert!(s.mercy_score >= 0.3);
        assert_eq!(c.tick_count, 1);
    }

    #[test]
    fn bless_registers() {
        let mut c = SimpleLatticeConductor::default();
        c.bless_system("mercy", 0.99, "phase1");
        assert!(c.registry.is_blessed("mercy"));
    }
}
