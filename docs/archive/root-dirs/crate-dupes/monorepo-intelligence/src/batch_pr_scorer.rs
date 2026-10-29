//! Automated scoring to help decide Focused vs Batch PR.
//! Implements the decision framework from docs/eternal-iteration-protocol.md

use std::collections::HashSet;

#[derive(Debug, Clone)]
pub struct ChangedFile {
    pub path: String,
    pub is_cross_crate: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrRecommendation {
    Focused,
    Batch { reason: String },
}

pub struct BatchPrScorer {
    files: Vec<ChangedFile>,
}

impl BatchPrScorer {
    pub fn new(files: Vec<ChangedFile>) -> Self {
        Self { files }
    }

    pub fn recommend(&self) -> PrRecommendation {
        let file_count = self.files.len();
        let cross_crate_count = self.files.iter().filter(|f| f.is_cross_crate).count();
        let unique_crates: HashSet<_> = self.files.iter()
            .filter_map(|f| f.path.split('/').nth(1))
            .collect();

        let cross_ratio = if file_count > 0 { cross_crate_count as f32 / file_count as f32 } else { 0.0 };

        let mut score = 0;
        if file_count >= 3 { score += 2; }
        if cross_ratio > 0.3 { score += 2; }
        if unique_crates.len() >= 2 { score += 2; }

        if score >= 4 {
            PrRecommendation::Batch {
                reason: format!("{} files across {} crates → Batch recommended", file_count, unique_crates.len()),
            }
        } else {
            PrRecommendation::Focused
        }
    }
}
