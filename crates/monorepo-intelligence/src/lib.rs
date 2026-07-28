pub mod config;
pub mod github;
pub mod health;
pub mod inheritance;
pub mod plugin;
pub mod predictive_coding;
pub mod report;
pub mod scanner;
pub mod search;

pub use inheritance::{analyze_inheritance, InheritanceStatus};
pub use predictive_coding::{
    ExpectedFreeEnergy, HierarchicalPredictiveCoding, PredictiveCodingError,
    PredictiveCodingResult, MERCY_VALENCE_FLOOR,
};
pub use scanner::{MonorepoScanner, ScanError, ScanResult, ScannedFile};

use std::path::PathBuf;

/// Main entry point for Ra-Thor Monorepo Intelligence.
/// Provides high-level access to scanning, inheritance analysis, health scoring,
/// hierarchical predictive coding / active inference, and reporting.
pub struct MonorepoIntelligence {
    pub root_path: PathBuf,
}

impl MonorepoIntelligence {
    pub fn new(root: impl AsRef<std::path::Path>) -> Self {
        Self {
            root_path: root.as_ref().to_path_buf(),
        }
    }

    /// Returns a scanner configured for this monorepo root.
    pub fn scanner(&self) -> MonorepoScanner {
        MonorepoScanner::new(self.root_path.clone())
    }

    /// Performs inheritance compliance analysis across all crates.
    pub fn analyze_inheritance(&self) -> Vec<InheritanceStatus> {
        analyze_inheritance(&self.root_path)
    }

    /// Runs a full scan and returns rich results.
    pub fn full_scan(&self) -> Result<ScanResult, ScanError> {
        self.scanner().scan()
    }

    /// Create a HierarchicalPredictiveCoding engine seeded with current lattice valence floor.
    pub fn predictive_engine(&self) -> HierarchicalPredictiveCoding {
        HierarchicalPredictiveCoding::new()
    }

    /// Convenience: run hierarchical predictive coding on a sensory / error signal.
    pub fn hierarchical_predictive_coding(
        &self,
        sensory_input: f64,
        requested_depth: u32,
    ) -> Result<PredictiveCodingResult, PredictiveCodingError> {
        self.predictive_engine()
            .hierarchical_predictive_coding(sensory_input, requested_depth)
    }

    /// Convenience: active inference free-energy minimization steps.
    pub fn active_inference(
        &self,
        prediction_error: f64,
        steps: u32,
    ) -> Result<f64, PredictiveCodingError> {
        self.predictive_engine()
            .integrate_with_active_inference_v2(prediction_error, steps)
    }
}
