//! Protocol Compliance Checker
//!
//! Provides automated checks against the Eternal Iteration Protocol.
//! This is the foundation for deeper custom code review automation (Option C).

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComplianceIssue {
    MissingThunderLockedIn,
    ShortPRBody,
    MissingCouncilReference,
    PotentialBatchCandidate { reason: String },
}

pub struct ProtocolComplianceChecker {
    pub min_body_length: usize,
}

impl Default for ProtocolComplianceChecker {
    fn default() -> Self {
        Self { min_body_length: 250 }
    }
}

impl ProtocolComplianceChecker {
    pub fn new(min_body_length: usize) -> Self {
        Self { min_body_length }
    }

    /// Check a PR body for basic protocol compliance.
    pub fn check_pr_body(&self, body: &str) -> Vec<ComplianceIssue> {
        let mut issues = Vec::new();

        if body.len() < self.min_body_length {
            issues.push(ComplianceIssue::ShortPRBody);
        }
        if !body.to_lowercase().contains("thunder locked in") {
            issues.push(ComplianceIssue::MissingThunderLockedIn);
        }
        if !body.to_lowercase().contains("patsagi") && !body.to_lowercase().contains("council") {
            issues.push(ComplianceIssue::MissingCouncilReference);
        }

        issues
    }

    /// Suggest whether this change might be better as a Batch PR.
    pub fn suggest_batch(&self, changed_file_count: usize, cross_crate: bool) -> Option<ComplianceIssue> {
        if changed_file_count >= 4 || cross_crate {
            Some(ComplianceIssue::PotentialBatchCandidate {
                reason: format!("{} files changed, cross-crate={}", changed_file_count, cross_crate),
            })
        } else {
            None
        }
    }
}
