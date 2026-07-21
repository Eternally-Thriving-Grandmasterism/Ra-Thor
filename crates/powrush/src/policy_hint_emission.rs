//! crates/powrush/src/policy_hint_emission.rs — v14.15.1
//! Ra-Thor Policy Hint Emission — sealed dual-repo RTT bridge to Powrush-MMO
//!
//! Canonical contract: Powrush-MMO/docs/RA_THOR_POLICY_HINT_EMISSION.md
//! Schema: ra_thor_policy_hint_v1
//! Governance: TOLC 8 Living Mercy Gates + PATSAGi Councils
//! Contact: info@Rathor.ai
//!
//! After a successful, mercy-passing deliberation on Powrush telemetry,
//! call `emit_policy_hints` (or the higher-level helper) to write soft,
//! non-authoritative recommendations that Powrush can apply.

use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;

/// Closed category set accepted by Powrush validation (zero-harm).
pub const CLOSED_CATEGORIES: &[&str] = &[
    "abundance_bias",
    "peaceful_resolution_weight",
    "ethical_floor",
    "council_participation_nudge",
    "innovation_encouragement",
    "mercy_presence",
];

/// Canonical schema identifier.
pub const SCHEMA_V1: &str = "ra_thor_policy_hint_v1";

/// Default emission path watched by Powrush PolicyHintInbox.
pub const DEFAULT_EMISSION_PATH: &str = "artifacts/ra_thor_policy_hints.json";

#[derive(Debug, Error)]
pub enum EmissionError {
    #[error("invalid category '{0}' — must be one of the closed set")]
    InvalidCategory(String),
    #[error("recommended_delta must be >= 0.0, got {0}")]
    NegativeDelta(f64),
    #[error("strength and mercy_factor must be in [0.0, 1.0]")]
    OutOfRangeFactor,
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON serialization error: {0}")]
    Serde(#[from] serde_json::Error),
    #[error("path parent missing for atomic write")]
    MissingParent,
}

/// Single soft policy hint.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PolicyHint {
    pub hint_id: String,
    pub category: String,
    pub strength: f64,
    pub mercy_factor: f64,
    pub recommended_delta: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rationale: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expires_at_unix: Option<u64>,
}

impl PolicyHint {
    /// Construct and validate a single hint.
    pub fn new(
        hint_id: impl Into<String>,
        category: impl Into<String>,
        strength: f64,
        mercy_factor: f64,
        recommended_delta: f64,
        rationale: Option<String>,
        expires_at_unix: Option<u64>,
    ) -> Result<Self, EmissionError> {
        let category = category.into();
        if !CLOSED_CATEGORIES.contains(&category.as_str()) {
            return Err(EmissionError::InvalidCategory(category));
        }
        if recommended_delta < 0.0 {
            return Err(EmissionError::NegativeDelta(recommended_delta));
        }
        if !(0.0..=1.0).contains(&strength) || !(0.0..=1.0).contains(&mercy_factor) {
            return Err(EmissionError::OutOfRangeFactor);
        }
        Ok(Self {
            hint_id: hint_id.into(),
            category,
            strength,
            mercy_factor,
            recommended_delta,
            rationale,
            expires_at_unix,
        })
    }
}

/// Full emission envelope matching the sealed contract.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PolicyHintEnvelope {
    pub schema: String,
    pub source: String,
    pub emitted_at_unix: u64,
    pub target_session_id: String,
    pub source_export_seq: u64,
    pub hints: Vec<PolicyHint>,
}

impl PolicyHintEnvelope {
    /// Build a validated envelope ready for emission.
    pub fn build(
        target_session_id: impl Into<String>,
        source_export_seq: u64,
        hints: Vec<PolicyHint>,
    ) -> Result<Self, EmissionError> {
        // Re-validate every hint (defense in depth)
        for h in &hints {
            if !CLOSED_CATEGORIES.contains(&h.category.as_str()) {
                return Err(EmissionError::InvalidCategory(h.category.clone()));
            }
            if h.recommended_delta < 0.0 {
                return Err(EmissionError::NegativeDelta(h.recommended_delta));
            }
            if !(0.0..=1.0).contains(&h.strength) || !(0.0..=1.0).contains(&h.mercy_factor) {
                return Err(EmissionError::OutOfRangeFactor);
            }
        }

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        Ok(Self {
            schema: SCHEMA_V1.to_string(),
            source: "ra-thor".to_string(),
            emitted_at_unix: now,
            target_session_id: target_session_id.into(),
            source_export_seq,
            hints,
        })
    }
}

/// Emit the envelope to the path watched by Powrush (atomic write).
///
/// Prefer calling this only after a successful, mercy-passing deliberation.
pub fn emit_policy_hints(
    envelope: &PolicyHintEnvelope,
    path: Option<&Path>,
) -> Result<PathBuf, EmissionError> {
    let target = path
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from(DEFAULT_EMISSION_PATH));

    // Ensure parent directory exists
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent)?;
    } else {
        return Err(EmissionError::MissingParent);
    }

    // Atomic write: temp file in same directory + rename
    let tmp = target.with_extension("json.tmp");
    {
        let mut f = fs::File::create(&tmp)?;
        let json = serde_json::to_string_pretty(envelope)?;
        f.write_all(json.as_bytes())?;
        f.sync_all()?;
    }
    fs::rename(&tmp, &target)?;

    Ok(target)
}

/// High-level helper: build + emit in one call after deliberation succeeds.
///
/// `session_id` and `export_seq` should come from the originating Powrush telemetry.
/// Prefer small, conservative deltas. Only emit when TOLC 8 / mercy gates pass.
pub fn emit_after_deliberation(
    session_id: impl Into<String>,
    export_seq: u64,
    hints: Vec<PolicyHint>,
    path: Option<&Path>,
) -> Result<PathBuf, EmissionError> {
    let envelope = PolicyHintEnvelope::build(session_id, export_seq, hints)?;
    emit_policy_hints(&envelope, path)
}

/// Convenience constructor for a conservative mercy-presence nudge.
pub fn conservative_mercy_presence(
    hint_id: impl Into<String>,
    strength: f64,
    delta: f64,
    rationale: impl Into<String>,
) -> Result<PolicyHint, EmissionError> {
    PolicyHint::new(
        hint_id,
        "mercy_presence",
        strength.clamp(0.0, 1.0),
        0.95, // high mercy floor
        delta.max(0.0),
        Some(rationale.into()),
        None,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::env;

    #[test]
    fn closed_set_rejects_unknown() {
        let err = PolicyHint::new("h1", "unknown_cat", 0.5, 0.9, 0.1, None, None);
        assert!(matches!(err, Err(EmissionError::InvalidCategory(_))));
    }

    #[test]
    fn negative_delta_rejected() {
        let err = PolicyHint::new("h1", "abundance_bias", 0.5, 0.9, -0.1, None, None);
        assert!(matches!(err, Err(EmissionError::NegativeDelta(_))));
    }

    #[test]
    fn valid_envelope_builds() {
        let h = PolicyHint::new(
            "test-hint-001",
            "mercy_presence",
            0.4,
            0.97,
            0.05,
            Some("PATSAGi consensus: gentle presence uplift".into()),
            None,
        )
        .unwrap();
        let env = PolicyHintEnvelope::build("session-abc", 42, vec![h]).unwrap();
        assert_eq!(env.schema, SCHEMA_V1);
        assert_eq!(env.source, "ra-thor");
        assert_eq!(env.hints.len(), 1);
    }

    #[test]
    fn atomic_emit_roundtrip() {
        // Pure-std temporary path under system temp
        let mut path = env::temp_dir();
        path.push(format!("ra_thor_policy_hints_test_{}.json", std::process::id()));

        let h = PolicyHint::new(
            "rt-001",
            "ethical_floor",
            0.6,
            0.98,
            0.03,
            Some("TOLC 8 ethical floor reinforcement".into()),
            None,
        )
        .unwrap();

        let written = emit_after_deliberation("sess-xyz", 7, vec![h], Some(&path)).unwrap();
        assert!(written.exists());

        let raw = fs::read_to_string(&written).unwrap();
        let parsed: PolicyHintEnvelope = serde_json::from_str(&raw).unwrap();
        assert_eq!(parsed.target_session_id, "sess-xyz");
        assert_eq!(parsed.source_export_seq, 7);
        assert_eq!(parsed.hints[0].category, "ethical_floor");

        // Cleanup
        let _ = fs::remove_file(&written);
    }
}
