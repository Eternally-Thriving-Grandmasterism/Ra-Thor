//! Conductable / MercyAligned — v13 integration traits (compat)
//!
//! Faithful reimplementation so leaf crates can implement the same contracts
//! against lattice-conductor-v14 + feature "v13-compat".

use crate::compat_v13::geometric::{GeometricState, MercyWeightedVote};
use std::collections::HashMap;

/// A system that can be conducted by the Lattice Conductor.
pub trait Conductable {
    fn system_id(&self) -> &'static str;
    fn system_name(&self) -> &'static str;
    fn on_conductor_tick(&mut self, conductor_state: &GeometricState);
    fn get_mercy_state(&self) -> Option<f64> {
        None
    }
}

/// Systems that participate in mercy-weighted coordination.
pub trait MercyAligned: Conductable {
    fn apply_mercy_influence(&mut self, vote: &MercyWeightedVote);
    fn current_mercy_score(&self) -> f64;
}

/// Formal blessing/registration record issued by the Conductor.
#[derive(Debug, Clone)]
pub struct SystemBlessing {
    pub system_id: String,
    pub blessed_at_tick: u64,
    pub mercy_alignment: f64,
    pub notes: String,
}

/// Lightweight registry for systems connected to the Conductor.
#[derive(Debug, Default)]
pub struct ConductorRegistry {
    pub registered_systems: HashMap<String, SystemBlessing>,
    pub tick_count: u64,
}

impl ConductorRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn bless_system(
        &mut self,
        system_id: &str,
        mercy_alignment: f64,
        notes: &str,
    ) -> SystemBlessing {
        let blessing = SystemBlessing {
            system_id: system_id.to_string(),
            blessed_at_tick: self.tick_count,
            mercy_alignment,
            notes: notes.to_string(),
        };
        self.registered_systems
            .insert(system_id.to_string(), blessing.clone());
        blessing
    }

    pub fn is_blessed(&self, system_id: &str) -> bool {
        self.registered_systems.contains_key(system_id)
    }

    pub fn get_blessing(&self, system_id: &str) -> Option<&SystemBlessing> {
        self.registered_systems.get(system_id)
    }

    pub fn advance_tick(&mut self) {
        self.tick_count = self.tick_count.saturating_add(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Dummy;

    impl Conductable for Dummy {
        fn system_id(&self) -> &'static str {
            "dummy"
        }
        fn system_name(&self) -> &'static str {
            "Dummy System"
        }
        fn on_conductor_tick(&mut self, _conductor_state: &GeometricState) {}
    }

    #[test]
    fn registry_bless_and_lookup() {
        let mut reg = ConductorRegistry::new();
        let b = reg.bless_system("dummy", 0.95, "test");
        assert_eq!(b.system_id, "dummy");
        assert!(reg.is_blessed("dummy"));
        assert!(!reg.is_blessed("missing"));
    }
}
