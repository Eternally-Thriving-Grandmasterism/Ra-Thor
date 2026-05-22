// ... existing code ...

impl MercyEvaluationHistory {
    // ... existing methods ...

    /// Detects potential system drift based on recent trends and council activity.
    /// Returns true if concerning patterns are observed.
    pub fn detect_drift(&self) -> bool {
        let declining_trends = matches!(self.mercy_compliance_trend, MetricTrend::Declining)
            || matches!(self.valence_level_trend, MetricTrend::Declining);

        let high_council_activity = self.recent_council_triggers > 2;

        // Drift is detected when we have declining trends combined with elevated council activity
        declining_trends && high_council_activity
    }

    /// Returns a severity level for detected drift (0.0 = none, 1.0 = high).
    pub fn drift_severity(&self) -> f64 {
        if !self.detect_drift() {
            return 0.0;
        }

        let trend_severity = match (self.mercy_compliance_trend, self.valence_level_trend) {
            (MetricTrend::Declining, MetricTrend::Declining) => 0.7,
            (MetricTrend::Declining, _) | (_, MetricTrend::Declining) => 0.5,
            _ => 0.3,
        };

        let council_severity = (self.recent_council_triggers as f64 / 5.0).min(0.5);

        (trend_severity + council_severity).min(1.0)
    }
}

// ... existing code ...