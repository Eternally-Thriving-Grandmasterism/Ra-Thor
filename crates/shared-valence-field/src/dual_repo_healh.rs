//! dual_repo_health.rs
//! Minimal dual-repo soft-feedback health check surface
//! Used by SharedValenceFieldGuard before flag activation
//! Phase B — Living Valence Organism
//! AG-SML v1.0 | Contact: info@Rathor.ai

/// Simple health report from the sealed dual-repo soft-feedback organism
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DualRepoHealth {
    pub ra_thor_healthy: bool,
    pub powrush_healthy: bool,
    pub feedback_loop_closed: bool,
}

impl DualRepoHealth {
    pub fn is_fully_healthy(&self) -> bool {
        self.ra_thor_healthy && self.powrush_healthy && self.feedback_loop_closed
    }
}

/// Placeholder checker (will later call the real sealed dual-repo health surface)
pub fn check_dual_repo_health() -> DualRepoHealth {
    // TODO: bind to actual dual-repo soft-feedback health endpoint
    DualRepoHealth {
        ra_thor_healthy: true,
        powrush_healthy: true,
        feedback_loop_closed: true,
    }
}
