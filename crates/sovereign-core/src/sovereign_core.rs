//! SovereignCore with full TuningTarget::ArbitrationInterval wiring + dynamic participation + metrics exposure

use mercy_gating_runtime::{
    PatsagiGovernance, CouncilArbitrationSession, PatsagiCouncil, TuningTarget,
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

            // Dynamic council participation based on current stake
            let mut session = CouncilArbitrationSession::new(current_turn);

            let mut sorted: Vec<_> = self.patsagi_governance.registry.stakes.iter().collect();
            sorted.sort_by_key(|(_, s)| std::cmp::Reverse(s.amount));

            for (id, stake) in sorted.iter().take(4) {
                session.add_council(PatsagiCouncil {
                    id: **id,
                    name: format!("Council-{} (stake:{}) ", id, stake.amount),
                });
            }

            if session.participating_councils.is_empty() {
                // Fallback
                session.add_council(PatsagiCouncil { id: 13, name: "Council of Coherence".into() });
                session.add_council(PatsagiCouncil { id: 24, name: "Council of Unity".into() });
            }

            let accepted = self.patsagi_governance.run_arbitration(&session, current_turn);
            self.metrics.proposals_accepted += accepted.len() as u64;

            if !accepted.is_empty() {
                let _ = self.lattice_conductor.hot_reload_mercy_parameters(&accepted);
            }

            self.last_arbitration_turn = current_turn;
        }

        self.lattice_conductor.run_eternal_cycle_production();
    }

    /// Fully wired entry point for TuningTarget::ArbitrationInterval
    pub fn apply_council_tuning(&mut self, target: &TuningTarget, value: f64) {
        if *target == TuningTarget::ArbitrationInterval {
            self.apply_arbitration_interval_tuning(value as u64);
        }
    }

    pub fn get_arbitration_metrics(&self) -> &ArbitrationMetrics {
        &self.metrics
    }

    fn should_run_arbitration(&self, current_turn: u64) -> bool {
        current_turn.saturating_sub(self.last_arbitration_turn) >= self.arbitration_interval
    }

    pub fn apply_arbitration_interval_tuning(&mut self, new_interval: u64) {
        if new_interval >= 1 {
            println!("[ARBITRATION] Interval tuned to {}", new_interval);
            self.arbitration_interval = new_interval;
        }
    }
}