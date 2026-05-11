pub mod audit_report;
pub mod mercy_metrics;
pub mod drift_report;

pub use audit_report::AuditReport;
pub use mercy_metrics::MercyMetrics;
pub use drift_report::DriftReport;

/// Main entry point for running a full monorepo audit.
pub fn run_full_audit() -> AuditReport {
    // Placeholder for now — will be expanded with real scanning logic
    AuditReport::new()
}