//! SovereignCore with full PatsagiGovernance integration

use mercy_gating_runtime::PatsagiGovernance;

pub struct SovereignCore {
    pub lattice_conductor: LatticeConductor,
    pub mercy_runtime: MercyGatingRuntime,
    pub patsagi_governance: PatsagiGovernance,
}

impl SovereignCore {
    pub fn new() -> Self {
        Self {
            lattice_conductor: LatticeConductor::new(),
            mercy_runtime: MercyGatingRuntime::new(),
            patsagi_governance: PatsagiGovernance::new(),
        }
    }

    pub fn run_council_arbitration(
        &mut self,
        session: &mercy_gating_runtime::CouncilArbitrationSession,
        current_turn: u64,
    ) -> Vec<mercy_gating_runtime::CouncilTuningProposal> {
        let accepted = self.patsagi_governance.run_arbitration(session, current_turn);

        if !accepted.is_empty() {
            let _ = self.lattice_conductor.hot_reload_mercy_parameters(&accepted);
        }

        accepted
    }
}