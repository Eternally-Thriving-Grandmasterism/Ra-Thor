//! SovereignCore with generated arbitration sessions, logging, and tunable interval

use mercy_gating_runtime::{
    PatsagiGovernance, CouncilArbitrationSession, PatsagiCouncil,
};

pub struct SovereignCore {
    pub lattice_conductor: LatticeConductor,
    pub mercy_runtime: MercyGatingRuntime,
    pub patsagi_governance: PatsagiGovernance,

    pub arbitration_interval: u64,
    pub last_arbitration_turn: u64,
    pub arbitration_events: u64,           // Metrics
}

impl SovereignCore {
    pub fn new() -> Self {
        Self {
            lattice_conductor: LatticeConductor::new(),
            mercy_runtime: MercyGatingRuntime::new(),
            patsagi_governance: PatsagiGovernance::new(),
            arbitration_interval: 50,
            last_arbitration_turn: 0,
            arbitration_events: 0,
        }
    }

    pub fn with_arbitration_interval(mut self, interval: u64) -> Self {
        self.arbitration_interval = interval;
        self
    }

    pub fn run_eternal_cycle_production(&mut self, current_turn: u64) {
        if self.should_run_arbitration(current_turn) {
            self.arbitration_events += 1;
            println!("[ARBITRATION] Window opened at turn {} (event #{}) ", current_turn, self.arbitration_events);

            // Generate a basic but real CouncilArbitrationSession
            let mut session = CouncilArbitrationSession::new(current_turn);
            session.add_council(PatsagiCouncil { id: 13, name: "Council of Coherence".into() });
            session.add_council(PatsagiCouncil { id: 24, name: "Council of Unity".into() });

            let accepted = self.patsagi_governance.run_arbitration(&session, current_turn);

            if !accepted.is_empty() {
                let _ = self.lattice_conductor.hot_reload_mercy_parameters(&accepted);
                println!("[ARBITRATION] {} proposals accepted and applied", accepted.len());
            } else {
                println!("[ARBITRATION] No proposals passed filters this cycle");
            }

            self.last_arbitration_turn = current_turn;
        }

        self.lattice_conductor.run_eternal_cycle_production();
    }

    fn should_run_arbitration(&self, current_turn: u64) -> bool {
        current_turn.saturating_sub(self.last_arbitration_turn) >= self.arbitration_interval
    }

    /// Allow dynamic tuning of arbitration interval via council proposals
    pub fn apply_arbitration_interval_tuning(&mut self, new_interval: u64) {
        if new_interval >= 1 {
            println!("[ARBITRATION] Interval tuned: {} -> {}", self.arbitration_interval, new_interval);
            self.arbitration_interval = new_interval;
        }
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