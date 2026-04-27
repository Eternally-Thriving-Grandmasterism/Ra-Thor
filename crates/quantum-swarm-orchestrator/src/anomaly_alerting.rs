//! # Anomaly Alerting for Ra-Thor Quantum Swarm
//!
//! **Production-grade real-time anomaly detection and alerting.**
//!
//! This module monitors the swarm for dangerous deviations and triggers
//! alerts at different severity levels. It is designed to integrate directly
//! with `SwarmMonitor` and can be extended to send notifications (email, Slack, etc.).

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Severity levels for alerts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertLevel {
    Info,
    Warning,
    Critical,
}

/// Types of anomalies the system can detect.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalyType {
    SuddenMercyDrop { previous: f64, current: f64 },
    GatePassRateCollapse { rate: f64 },
    HebbianBondWeakening { average: f64 },
    GammaDegradation { current: f64 },
    StagnantCEHI { days: u32 },
    PlanetaryScaleInstability { effective_gamma: f64 },
}

/// A single alert event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    pub timestamp: DateTime<Utc>,
    pub level: AlertLevel,
    pub anomaly: AnomalyType,
    pub message: String,
    pub recommended_action: String,
}

/// Configurable thresholds for anomaly detection.
#[derive(Debug, Clone)]
pub struct AlertConfig {
    pub mercy_drop_threshold: f64,      // e.g. 0.05 (5% drop)
    pub gate_pass_critical: f64,        // e.g. 0.75
    pub hebbian_weak_threshold: f64,    // e.g. 0.65
    pub gamma_degradation_threshold: f64, // e.g. 0.0025
    pub stagnant_cehi_days: u32,        // e.g. 14
}

impl Default for AlertConfig {
    fn default() -> Self {
        Self {
            mercy_drop_threshold: 0.05,
            gate_pass_critical: 0.75,
            hebbian_weak_threshold: 0.65,
            gamma_degradation_threshold: 0.0025,
            stagnant_cehi_days: 14,
        }
    }
}

/// Real-time anomaly detector.
pub struct AnomalyDetector {
    config: AlertConfig,
    previous_mercy: Option<f64>,
    stagnant_counter: u32,
}

impl AnomalyDetector {
    pub fn new(config: AlertConfig) -> Self {
        Self {
            config,
            previous_mercy: None,
            stagnant_counter: 0,
        }
    }

    /// Analyzes the latest snapshot and returns any alerts.
    pub fn analyze(&mut self, snapshot: &super::real_time_swarm_monitoring::SwarmSnapshot) -> Vec<Alert> {
        let mut alerts = Vec::new();

        // 1. Sudden Mercy Drop
        if let Some(prev) = self.previous_mercy {
            let drop = prev - snapshot.mercy_valence;
            if drop > self.config.mercy_drop_threshold {
                alerts.push(Alert {
                    timestamp: snapshot.timestamp,
                    level: AlertLevel::Warning,
                    anomaly: AnomalyType::SuddenMercyDrop { previous: prev, current: snapshot.mercy_valence },
                    message: format!("Sudden mercy-valence drop of {:.1}%", drop * 100.0),
                    recommended_action: "Increase daily TOLC practice intensity immediately.".to_string(),
                });
            }
        }
        self.previous_mercy = Some(snapshot.mercy_valence);

        // 2. Gate Pass Rate Collapse
        if snapshot.gate_pass_rate < self.config.gate_pass_critical {
            alerts.push(Alert {
                timestamp: snapshot.timestamp,
                level: AlertLevel::Critical,
                anomaly: AnomalyType::GatePassRateCollapse { rate: snapshot.gate_pass_rate },
                message: format!("Gate pass rate critically low: {:.1}%", snapshot.gate_pass_rate * 100.0),
                recommended_action: "Activate emergency TOLC protocols and GroupCollective sessions.".to_string(),
            });
        }

        // 3. Hebbian Bond Weakening
        if snapshot.avg_hebbian_bond < self.config.hebbian_weak_threshold {
            alerts.push(Alert {
                timestamp: snapshot.timestamp,
                level: AlertLevel::Warning,
                anomaly: AnomalyType::HebbianBondWeakening { average: snapshot.avg_hebbian_bond },
                message: format!("Average Hebbian bond weakening: {:.3}", snapshot.avg_hebbian_bond),
                recommended_action: "Focus on warm touch and GroupCollective practices.".to_string(),
            });
        }

        // 4. Gamma Degradation
        if snapshot.effective_gamma < self.config.gamma_degradation_threshold {
            alerts.push(Alert {
                timestamp: snapshot.timestamp,
                level: AlertLevel::Critical,
                anomaly: AnomalyType::GammaDegradation { current: snapshot.effective_gamma },
                message: format!("Convergence rate critically low: {:.5}", snapshot.effective_gamma),
                recommended_action: "Review 7-Gate compliance and increase high-joy practice days.".to_string(),
            });
        }

        // 5. Stagnant CEHI (simplified counter)
        if snapshot.collective_cehi < 4.0 {
            self.stagnant_counter += 1;
            if self.stagnant_counter >= self.config.stagnant_cehi_days {
                alerts.push(Alert {
                    timestamp: snapshot.timestamp,
                    level: AlertLevel::Warning,
                    anomaly: AnomalyType::StagnantCEHI { days: self.stagnant_counter },
                    message: format!("CEHI stagnant for {} days", self.stagnant_counter),
                    recommended_action: "Introduce new high-joy protocols and review daily practice quality.".to_string(),
                });
            }
        } else {
            self.stagnant_counter = 0;
        }

        alerts
    }

    /// Prints alerts to console with color-coded severity.
    pub fn print_alerts(&self, alerts: &[Alert]) {
        for alert in alerts {
            let prefix = match alert.level {
                AlertLevel::Info => "ℹ️",
                AlertLevel::Warning => "⚠️",
                AlertLevel::Critical => "🚨",
            };
            println!("{} [{}] {}", prefix, alert.timestamp.format("%H:%M:%S"), alert.message);
            println!("   Recommended: {}", alert.recommended_action);
        }
    }
}
