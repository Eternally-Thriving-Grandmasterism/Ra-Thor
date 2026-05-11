//! Plasticity Health Metrics
//!
//! Provides observability into how effective the plasticity rules are performing
//! over time. This allows ra-thor-meta-intelligence to learn which rules
//! are most effective and adjust strategy accordingly.

use serde::{Deserialize, Serialize};

/// Tracks the health and effectiveness of plasticity applications.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PlasticityHealthMetrics {
    /// Total number of plasticity applications attempted
    pub total_applications: u64,
    /// Number of successful applications (no rollback)
    pub successful_applications: u64,
    /// Number of rollbacks triggered
    pub rollbacks: u64,
    /// Average mercy impact before application
    pub avg_mercy_impact_before: f64,
    /// Average mercy impact after application
    pub avg_mercy_impact_after: f64,
    /// Most frequently used plasticity rule type
    pub most_used_rule: Option<String>,
    /// Simple trend indicator ("improving", "stable", "degrading")
    pub recent_trend: String,
}

impl PlasticityHealthMetrics {
    pub fn new() -> Self {
        Self {
            recent_trend: "stable".to_string(),
            ..Default::default()
        }
    }

    /// Record a plasticity application result
    pub fn record_application(
        &mut self,
        mercy_before: f64,
        mercy_after: f64,
        was_rollback: bool,
        rule_used: &str,
    ) {
        self.total_applications += 1;

        if was_rollback {
            self.rollbacks += 1;
        } else {
            self.successful_applications += 1;
        }

        // Update running averages (simple exponential moving average)
        let alpha = 0.2;
        self.avg_mercy_impact_before =
            alpha * mercy_before + (1.0 - alpha) * self.avg_mercy_impact_before;
        self.avg_mercy_impact_after =
            alpha * mercy_after + (1.0 - alpha) * self.avg_mercy_impact_after;

        // Track most used rule (simple counter would be better in future)
        self.most_used_rule = Some(rule_used.to_string());

        // Update trend
        let mercy_delta = mercy_after - mercy_before;
        if mercy_delta > 0.01 {
            self.recent_trend = "improving".to_string();
        } else if mercy_delta < -0.01 {
            self.recent_trend = "degrading".to_string();
        } else {
            self.recent_trend = "stable".to_string();
        }
    }

    pub fn success_rate(&self) -> f64 {
        if self.total_applications == 0 {
            return 1.0;
        }
        self.successful_applications as f64 / self.total_applications as f64
    }

    pub fn rollback_rate(&self) -> f64 {
        if self.total_applications == 0 {
            return 0.0;
        }
        self.rollbacks as f64 / self.total_applications as f64
    }
}
