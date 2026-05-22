// ... existing code above ...

// ==================== MA'AT KPI STRUCTURES (for SixteenMaat Level) ====================

/// The four core dimensions of Ma'at
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MaatDimension {
    Balance,
    Truth,
    Justice,
    Order,
}

/// Ma'at Key Performance Indicator for a specific gate or overall evaluation
#[derive(Debug, Clone)]
pub struct MaatKpi {
    pub dimension_scores: HashMap<MaatDimension, f64>,
}

impl Default for MaatKpi {
    fn default() -> Self {
        let mut scores = HashMap::new();
        scores.insert(MaatDimension::Balance, 0.0);
        scores.insert(MaatDimension::Truth, 0.0);
        scores.insert(MaatDimension::Justice, 0.0);
        scores.insert(MaatDimension::Order, 0.0);
        Self { dimension_scores: scores }
    }
}

impl MaatKpi {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_score(&mut self, dimension: MaatDimension, score: f64) {
        self.dimension_scores.insert(dimension, score.clamp(0.0, 1.0));
    }

    /// Returns the overall Ma'at score (average of all dimensions)
    pub fn overall_score(&self) -> f64 {
        if self.dimension_scores.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.dimension_scores.values().sum();
        sum / self.dimension_scores.len() as f64
    }

    /// Checks against common Ma'at thresholds (inspired by existing Ra-Thor documents)
    pub fn meets_threshold(&self, threshold: f64) -> bool {
        self.overall_score() >= threshold
    }
}

/// Extended verdict that can include Ma'at scoring
#[derive(Debug, Clone)]
pub enum ExtendedMercyVerdict {
    Passed { overall_score: f64, maat_kpi: Option<MaatKpi> },
    Mitigated { overall_score: f64, notes: Vec<String>, maat_kpi: Option<MaatKpi> },
    RequiresCouncilReview,
    Blocked { reason: String },
}

// ... existing code continues ...