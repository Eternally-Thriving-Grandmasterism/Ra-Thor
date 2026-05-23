//! Sovereign Core — integrates PATSAGi Council arbitration + staking

use mercy_gating_runtime::{
    CouncilArbitrationSession, CouncilStake, MercyGatingRuntime,
};
use std::collections::HashMap;
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

    /// Apply only proposals that reached both consensus AND have sufficient stake.
    pub fn apply_arbitration_session(
        &mut self,
        session: &CouncilArbitrationSession,
        stakes: &HashMap<u32, CouncilStake>,
    ) {
        let accepted = session.accepted_proposals_with_staking(stakes);

        if accepted.is_empty() {
            println!("[SOVEREIGN] No proposals passed consensus + stake requirements.");
            return;
        }

        println!("[SOVEREIGN] Applying {} staked + consensus-backed proposals", accepted.len());
        let _ = self.lattice_conductor.hot_reload_mercy_parameters(&accepted);
    }

    pub fn run_eternal_cycle_with_arbitration(
        &mut self,
        session: &CouncilArbitrationSession,
        stakes: &HashMap<u32, CouncilStake>,
    ) {
        self.apply_arbitration_session(session, stakes);
        self.lattice_conductor.run_eternal_cycle_production();
    }
}