//! ... existing code ...

#[derive(Debug, Clone, PartialEq)]
pub enum MercyVerdict {
    Passed { overall_score: f64 },
    Mitigated { overall_score: f64, notes: Vec<String> },
    RequiresCouncilReview,
    Blocked { reason: String },
}

// Added Eq manually where needed for simplicity

// ... rest of the file ...