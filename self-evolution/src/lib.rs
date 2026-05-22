// ... existing code in SovereignHealthMonitor ...

    /// Historical context for mercy and reflexion evaluations.
    /// This will be used by SovereignReflexionCore.
    pub mercy_history: crate::sovereign_reflexion_core::MercyEvaluationHistory,

    pub fn new() -> Self {
        Self {
            metrics: SovereignHealthMetrics::default(),
            blessing_history: Vec::new(),
            council_review_trigger_count: 0,
            mercy_history: crate::sovereign_reflexion_core::MercyEvaluationHistory::new(20),
        }
    }

// ... existing code ...