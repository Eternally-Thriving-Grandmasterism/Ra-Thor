//! SovereignCore with council governance as default path + periodic scheduling

use mercy_gating_runtime::PatsagiGovernance;

pub struct SovereignCore {
    pub lattice_conductor: LatticeConductor,
    pub mercy_runtime: MercyGatingRuntime,
    pub patsagi_governance: PatsagiGovernance,

    // Arbitration scheduling
    pub arbitration_interval: u64,      // Run council sessions every N turns
    pub last_arbitration_turn: u64,
}

impl SovereignCore {
    pub fn new() -> Self {
        Self {
            lattice_conductor: LatticeConductor::new(),
            mercy_runtime: MercyGatingRuntime::new(),
            patsagi_governance: PatsagiGovernance::new(),
            arbitration_interval: 50,
            last_arbitration_turn: 0,
        }
    }

    pub fn with_arbitration_interval(mut self, interval: u64) -> Self {
        self.arbitration_interval = interval;
        self
    }

    /// Default eternal cycle production now includes periodic council governance check
    pub fn run_eternal_cycle_production(&mut self, current_turn: u64) {
        if self.should_run_arbitration(current_turn) {
            println!("[SOVEREIGN] Council arbitration window open at turn {}", current_turn);
            self.last_arbitration_turn = current_turn;

            // In full implementation: create/retrieve a CouncilArbitrationSession and run it
            // let accepted = self.patsagi_governance.run_arbitration(&session, current_turn);
            // if !accepted.is_empty() { ... hot reload ... }
        }

        self.lattice_conductor.run_eternal_cycle_production();
    }

    fn should_run_arbitration(&self, current_turn: u64) -> bool {
        current_turn.saturating_sub(self.last_arbitration_turn) >= self.arbitration_interval
    }

    pub fn run_eternal_cycle_with_council_governance(
        &mut self,
        session: Option<&mercy_gating_runtime::CouncilArbitrationSession>,
        current_turn: u64,
    ) {
        if let Some(session) = session {
            let accepted = self.patsagi_governance.run_arbitration(session, current_turn);
            if !accepted.is_empty() {
                let _ = self.lattice_conductor.hot_reload_mercy_parameters(&accepted);
            }
        }

        self.lattice_conductor.run_eternal_cycle_production();
    }
}