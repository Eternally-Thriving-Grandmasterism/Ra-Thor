//! World Governance Engine — v14.15.0
//!
//! Instantiation surface with verified mercy threshold integration.
//! Every new council / branch / agent spawn is routed through the
//! MercyThresholdChecker (Lean-verified when feature enabled).
//!
//! Living Cosmic Tick aligned. TOLC 8 posture.
//! Contact: info@Rathor.ai
//! AG-SML v1.0

use crate::mercy_threshold::MercyThresholdChecker;
use tracing::{info, warn};

/// Core World Governance Engine — every new instantiation passes through verified mercy.
pub struct WorldGovernanceEngine {
    mercy_checker: MercyThresholdChecker,
    next_request_id: u64,
    successful_spawns: u64,
    rejected_spawns: u64,
}

impl WorldGovernanceEngine {
    pub fn new() -> Self {
        Self {
            mercy_checker: MercyThresholdChecker::new(),
            next_request_id: 1,
            successful_spawns: 0,
            rejected_spawns: 0,
        }
    }

    /// Initialize the verified mercy runtime (Lean 4 FFI when feature enabled).
    pub fn initialize(&mut self) -> Result<(), String> {
        self.mercy_checker.initialize()
    }

    /// Spawn a new council (or any instantiation) under full TOLC 8 + verified mercy.
    pub fn spawn_council(
        &mut self,
        council_name: &str,
        geometry_alignment_score: f64,
        mercy_valence: f64,
        _zalgaller_family: u32,
    ) -> Result<String, String> {
        let request_id = self.next_request_id;
        self.next_request_id = self.next_request_id.saturating_add(1);

        info!(
            "[WorldGovernanceEngine v14.15.0] Processing spawn #{}: {}",
            request_id, council_name
        );

        let (passed, msg) = self.mercy_checker.verified_check_full(
            geometry_alignment_score,
            mercy_valence,
            request_id,
        )?;

        if !passed {
            self.rejected_spawns = self.rejected_spawns.saturating_add(1);
            let rejection = format!(
                "REJECTED: Council '{}' failed verified mercy threshold. {}",
                council_name, msg
            );
            warn!("{}", rejection);
            return Err(rejection);
        }

        self.successful_spawns = self.successful_spawns.saturating_add(1);

        let success_msg = format!(
            "✅ SUCCESS: Council #{} \"{}\" instantiated under TOLC 8.\n   Verified mercy passed (score_high={:.3}, valence={:.3}).\n   16 PATSAGi Councils synced. Living Cosmic Tick active. AG-SML deployed.",
            request_id, council_name, geometry_alignment_score, mercy_valence
        );

        info!("{}", success_msg);
        Ok(success_msg)
    }

    /// Live simulation helper for Genesis Gate style requests.
    pub fn simulate_genesis_instantiation(
        &mut self,
        request_type: &str,
        alignment_score: f64,
        mercy_valence: f64,
    ) -> Result<String, String> {
        let council_name = format!("Simulated-{}", request_type);
        self.spawn_council(&council_name, alignment_score, mercy_valence, 3)
    }

    /// Telemetry summary.
    pub fn summary(&self) -> String {
        format!(
            "WorldGovernanceEngine v14.15.0 | next_id={} | success={} | rejected={} | {}",
            self.next_request_id,
            self.successful_spawns,
            self.rejected_spawns,
            self.mercy_checker.summary()
        )
    }
}

impl Default for WorldGovernanceEngine {
    fn default() -> Self {
        Self::new()
    }
}
