//! FENCA content audit.
//!
//! Four deterministic mercy checks. `passed` is true only when every check
//! passes. `audit_score` is the fraction of those checks that passed.
//! The score is not a valence and not a fixed baseline.
//!
//! A failed audit is unverified: `is_verified()` returns false and `reasons`
//! lists what failed. This crate does not certify a result.

use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::info;

/// Byte cap for one audit input. Longer text fails `within_size_cap`.
pub const MAX_AUDIT_BYTES: usize = 65_536;

const CHECK_NAMES: [&str; 4] = [
    "non_empty",
    "within_size_cap",
    "sources_when_required",
    "no_banned_overclaim",
];

/// Substrings that turn on the citation check. ASCII case-insensitive.
const SOURCES_REQUIRED_MARKERS: [&str; 3] =
    ["sources-required", "sources_required", "sources required"];

/// Overclaim substrings. The first column is matched on an ASCII-lowercase copy.
const BANNED_OVERCLAIM_MARKERS: [(&str, &str); 3] = [
    ("0.999999", "0.999999"),
    ("agsi achieved", "AGSi achieved"),
    ("perfect valence", "perfect valence"),
];

#[derive(Error, Debug)]
pub enum FencaError {
    #[error("FENCA veto — integrity check failed: {0}")]
    Veto(String),
    #[error("Internal audit error: {0}")]
    AuditError(String),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FencaAuditReport {
    pub passed: bool,
    pub audit_score: f64,
    pub checks_performed: Vec<String>,
    pub issues: Vec<String>,
    /// Why the audit failed. Empty when `passed` is true. Mirrored in `issues`.
    #[serde(default)]
    pub reasons: Vec<String>,
}

impl FencaAuditReport {
    /// False when any mercy check failed. That result is unverified.
    pub fn is_verified(&self) -> bool {
        self.passed
    }

    /// Fraction of checks that passed. Same value as `audit_score`.
    pub fn fidelity(&self) -> f64 {
        self.audit_score
    }
}

#[derive(Debug)]
pub struct FencaEternalCheck;

impl FencaEternalCheck {
    pub fn new() -> Self {
        Self
    }

    pub async fn perform_eternal_check(&self, input: &str) -> Result<FencaAuditReport, FencaError> {
        Ok(audit(input))
    }

    /// Archive entry: verify text and return a report that may be unverified.
    pub async fn verify_codex_content(content: &str) -> FencaAuditReport {
        audit(content)
    }

    /// Caller shape used by the council wasm surface.
    ///
    /// `scope` is accepted and ignored. It does not change the checks.
    /// Check failure is `Ok(false)`, not `Err`.
    pub async fn run_full_eternal_check(input: &str, _scope: &str) -> Result<bool, FencaError> {
        Ok(audit(input).passed)
    }
}

struct Check {
    passed: bool,
    reasons: Vec<String>,
}

fn audit(input: &str) -> FencaAuditReport {
    info!("fenca audit bytes={}", input.len());
    let folded = input.to_ascii_lowercase();
    let checks = [
        check_non_empty(input),
        check_size(input),
        check_sources(&folded),
        check_banned(&folded),
    ];
    debug_assert_eq!(checks.len(), CHECK_NAMES.len());

    let passed_count = checks.iter().filter(|check| check.passed).count();
    let reasons: Vec<String> = checks
        .iter()
        .flat_map(|check| check.reasons.iter().cloned())
        .collect();
    let passed = checks.iter().all(|check| check.passed);
    debug_assert_eq!(passed, reasons.is_empty());

    FencaAuditReport {
        passed,
        audit_score: passed_count as f64 / checks.len() as f64,
        checks_performed: CHECK_NAMES.iter().map(|name| (*name).to_string()).collect(),
        issues: reasons.clone(),
        reasons,
    }
}

fn check_non_empty(input: &str) -> Check {
    if input.trim().is_empty() {
        Check {
            passed: false,
            reasons: vec!["rejected: empty or whitespace-only input".to_string()],
        }
    } else {
        Check {
            passed: true,
            reasons: Vec::new(),
        }
    }
}

fn check_size(input: &str) -> Check {
    if input.len() > MAX_AUDIT_BYTES {
        Check {
            passed: false,
            reasons: vec![format!(
                "rejected: input is {} bytes, over the {MAX_AUDIT_BYTES} byte cap",
                input.len()
            )],
        }
    } else {
        Check {
            passed: true,
            reasons: Vec::new(),
        }
    }
}

fn check_sources(folded: &str) -> Check {
    if !sources_are_required(folded) || has_citation(folded) {
        return Check {
            passed: true,
            reasons: Vec::new(),
        };
    }
    Check {
        passed: false,
        reasons: vec!["rejected: sources are required and no citation was found".to_string()],
    }
}

fn sources_are_required(folded: &str) -> bool {
    SOURCES_REQUIRED_MARKERS
        .iter()
        .any(|marker| folded.contains(marker))
}

/// URL, DOI, a `[source:` tag, or a line that starts with `source:` or `citation:`.
fn has_citation(folded: &str) -> bool {
    if folded.contains("https://")
        || folded.contains("http://")
        || folded.contains("doi:")
        || folded.contains("[source:")
    {
        return true;
    }
    folded.lines().any(|line| {
        let line = line.trim();
        line.starts_with("source:") || line.starts_with("citation:")
    })
}

fn check_banned(folded: &str) -> Check {
    let reasons = BANNED_OVERCLAIM_MARKERS
        .iter()
        .filter(|(needle, _)| folded.contains(needle))
        .map(|(_, label)| format!("rejected: banned overclaim marker '{label}'"))
        .collect::<Vec<_>>();
    Check {
        passed: reasons.is_empty(),
        reasons,
    }
}
