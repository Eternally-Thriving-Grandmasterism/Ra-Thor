// ... existing code ...

/// Represents the directional trend of a metric over recent observations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MetricTrend {
    Improving,
    Stable,
    Declining,
    #[default]
    Unknown,
}

/// Maintains lightweight historical context for mercy and self-referential evaluations.
/// Tracks recent verdicts, council escalations, and metric trajectories.
#[derive(Debug, Clone)]
pub struct MercyEvaluationHistory {
    pub recent_verdicts: Vec<mercy_gating::MercyVerdict>,
    pub recent_council_triggers: u32,
    pub mercy_compliance_trend: MetricTrend,
    pub valence_level_trend: MetricTrend,
    max_history_size: usize,
}

impl MercyEvaluationHistory {
    pub fn new(max_size: usize) -> Self {
        Self {
            recent_verdicts: Vec::with_capacity(max_size),
            recent_council_triggers: 0,
            mercy_compliance_trend: MetricTrend::Unknown,
            valence_level_trend: MetricTrend::Unknown,
            max_history_size: max_size,
        }
    }

    /// Records a new mercy verdict and updates related counters.
    pub fn record_verdict(&mut self, verdict: mercy_gating::MercyVerdict) {
        if self.recent_verdicts.len() >= self.max_history_size {
            self.recent_verdicts.remove(0);
        }
        self.recent_verdicts.push(verdict.clone());

        if matches!(verdict, mercy_gating::MercyVerdict::RequiresCouncilReview) {
            self.recent_council_triggers += 1;
        }
    }

    /// Updates metric trend indicators based on previous and current values.
    pub fn update_trends(&mut self, previous_compliance: f64, current_compliance: f64,
                           previous_valence: f64, current_valence: f64) {
        self.mercy_compliance_trend = Self::calculate_trend(previous_compliance, current_compliance);
        self.valence_level_trend = Self::calculate_trend(previous_valence, current_valence);
    }

    fn calculate_trend(previous: f64, current: f64) -> MetricTrend {
        let delta = current - previous;
        if delta > 0.01 {
            MetricTrend::Improving
        } else if delta < -0.01 {
            MetricTrend::Declining
        } else {
            MetricTrend::Stable
        }
    }

    /// Returns a combined coherence signal influenced by recent verdicts and metric trends.
    pub fn coherence_signal(&self) -> f64 {
        let verdict_signal = if self.recent_verdicts.is_empty() {
            0.7
        } else {
            let positive = self.recent_verdicts.iter().filter(|v| {
                matches!(v, mercy_gating::MercyVerdict::Passed { .. }
                    | mercy_gating::MercyVerdict::Mitigated { .. })
            }).count();
            positive as f64 / self.recent_verdicts.len() as f64
        };

        let trend_bonus = match (self.mercy_compliance_trend, self.valence_level_trend) {
            (MetricTrend::Improving, MetricTrend::Improving) => 0.08,
            (MetricTrend::Improving, _) | (_, MetricTrend::Improving) => 0.04,
            (MetricTrend::Declining, MetricTrend::Declining) => -0.06,
            _ => 0.0,
        };

        (verdict_signal + trend_bonus).clamp(0.0, 1.0)
    }

    /// Detects potential system drift based on recent trends and council activity.
    pub fn detect_drift(&self) -> bool {
        let declining_trends = matches!(self.mercy_compliance_trend, MetricTrend::Declining)
            || matches!(self.valence_level_trend, MetricTrend::Declining);

        let high_council_activity = self.recent_council_triggers > 2;
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