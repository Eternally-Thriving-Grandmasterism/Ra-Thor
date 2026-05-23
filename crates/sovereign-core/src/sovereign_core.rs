//! Sovereign Core — integrates PATSAGi Council arbitration
//! for dynamic, consensus-driven mercy tuning.

use mercy_gating_runtime::{
    CouncilArbitrationSession, MercyGatingRuntime,
};
use lattice_conductor_v13::LatticeConductor;

pub struct SovereignCore {
    pub lattice_conductor: LatticeConductor,
    pub mercy_runtime: MercyGatingRuntime,
}

impl SovereignCore {
    pub fn new() -> Self {
        Self {
            lattice_conductor: LatticeConductor::new(),
            mercy_runtime: MercyGatingRuntime::new(),
        }
    }

    /// Apply proposals from a PATSAGi arbitration session.
    /// Only proposals that reached consensus are applied via hot-reload.
    pub fn apply_arbitration_session(&mut self, session: &CouncilArbitrationSession) {
        if session.has_consensus() {
            let accepted = session.accepted_proposals();
            println!("[SOVEREIGN] Applying {} consensus-backed proposals (turn {})", accepted.len(), session.turn);

            let _ = self.lattice_conductor.hot_reload_mercy_parameters(&accepted);
        } else {
            println!("[SOVEREIGN] Arbitration did not reach consensus — no tuning applied.");
        }
    }

    /// Full eternal cycle step that includes council arbitration.
    pub fn run_eternal_cycle_with_arbitration(&mut self, session: &CouncilArbitrationSession) {
        self.apply_arbitration_session(session);
        self.lattice_conductor.run_eternal_cycle_production();
    }
}