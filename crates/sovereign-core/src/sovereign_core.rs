//! SovereignCore with richer arbitration sessions and proper metrics

use mercy_gating_runtime::{
    PatsagiGovernance, CouncilArbitrationSession, PatsagiCouncil,
};

#[derive(Debug, Default)]
pub struct ArbitrationMetrics {
    pub total_events: u64,
    pub proposals_accepted: u64,
    pub proposals_slashed: u64,
    pub last_event_turn: u64,
}

pub struct SovereignCore {
    pub lattice_conductor: LatticeConductor,
    pub mercy_runtime: MercyGatingRuntime,
    pub patsagi_governance: PatsagiGovernance,

    pub arbitration_interval: u64,
    pub last_arbitration_turn: u64,
    pub metrics: ArbitrationMetrics,
}

impl SovereignCore {
    pub fn new() -> Self {
        Self {
            lattice_conductor: LatticeConductor::new(),
            mercy_runtime: MercyGatingRuntime::new(),
            patsagi_governance: PatsagiGovernance::new(),
            arbitration_interval: 50,
            last_arbitration_turn: 0,
            metrics: ArbitrationMetrics::default(),
        }
    }

    pub fn run_eternal_cycle_production(&mut self, current_turn: u64) {
        if self.should_run_arbitration(current_turn) {
            self.metrics.total_events += 1;
            self.metrics.last_event_turn = current_turn;

            println!("[ARBITRATION] Event #{} at turn {}", self.metrics.total_events, current_turn);

            // Richer generated session with more councils
            let mut session = CouncilArbitrationSession::new(current_turn);
            session.add_council(PatsagiCouncil { id: 7,  name: "Council of Harmony".into() });
            session.add_council(PatsagiCouncil { id: 13, name: "Council of Coherence".into() });
            session.add_council(PatsagiCouncil { id: 17, name: "Council of Truth".into() });
            session.add_council(PatsagiCouncil { id: 24, name: "Council of Unity".into() });

            let accepted = self.patsagi_governance.run_arbitration(&session, current_turn);
            self.metrics.proposals_accepted += accepted.len() as u64;

            if !accepted.is_empty() {
                let _ = self.lattice_conductor.hot_reload_mercy_parameters(&accepted);
                println!("[ARBITRATION] {} proposals accepted", accepted.len());
            }

            self.last_arbitration_turn = current_turn;
        }

        self.lattice_conductor.run_eternal_cycle_production();
    }

    fn should_run_arbitration(&self, current_turn: u64) -> bool {
        current_turn.saturating_sub(self.last_arbitration_turn) >= self.arbitration_interval
    }

    pub fn apply_arbitration_interval_tuning(&mut self, new_interval: u64) {
        if new_interval >= 1 {
            println!("[ARBITRATION] Interval tuned {} -> {}", self.arbitration_interval, new_interval);
            self.arbitration_interval = new_interval;
        }
    }
}