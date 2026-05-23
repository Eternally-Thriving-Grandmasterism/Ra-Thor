//! SovereignCore with full TuningTarget::ArbitrationInterval support + metrics exposure

use mercy_gating_runtime::TuningTarget;

// ... (ArbitrationMetrics and other structs remain) ...

impl SovereignCore {

    /// Full proposal handling for TuningTarget::ArbitrationInterval
    pub fn handle_proposal(&mut self, target: &TuningTarget, value: f64) {
        match target {
            TuningTarget::ArbitrationInterval => {
                self.apply_arbitration_interval_tuning(value as u64);
            }
            _ => {}
        }
    }

    /// Simple logging hook to expose metrics (can be called periodically)
    pub fn log_metrics(&self) {
        println!("[METRICS] Arbitration Events: {} | Proposals Accepted: {} | Last Turn: {}",
            self.metrics.total_events,
            self.metrics.proposals_accepted,
            self.metrics.last_event_turn
        );
    }

    // Existing methods (run_eternal_cycle_production, get_arbitration_metrics, etc.) remain
}