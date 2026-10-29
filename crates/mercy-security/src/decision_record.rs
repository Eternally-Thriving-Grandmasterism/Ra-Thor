//! Replayable Layer 0 admission decision record.
//!
//! Serde JSON log for admit / block / override. Hash of the ingest only.
//! Never stores raw payload bytes. Does **not** change `admit_or_block`
//! thresholds or keyword tables.
//!
//! Not a hash-chained SIEM. Not EU AI Act logging. Not METR.
//! Contact: info@Rathor.ai

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;
use uuid::Uuid;

use crate::{IngestionScanResult, IngestionScanner, IngestionThreat, MercySecurityError, RiskTier};

/// Living surface id. Deserialize rejects any other string.
pub const DECISION_RECORD_SURFACE: &str = "layer0-admission-shell";

/// Label of the existing unattended policy: None/Low admit, Medium+ block, >4 MiB too large.
/// Not a new threshold. Do not bump this string when only the record schema changes.
pub const DECISION_RECORD_POLICY_VERSION: &str = "admit_or_block.none_low.v1";

const FORBIDDEN_LOG_KEYS: &[&str] = &[
    "payload",
    "raw_payload",
    "content",
    "body",
    "bytes",
    "raw",
    "ingest",
    "text",
];

const ALLOWED_LOG_KEYS: &[&str] = &[
    "record_id",
    "timestamp_utc",
    "policy_version",
    "crate_version",
    "surface",
    "verdict",
    "threat_class",
    "reason_codes",
    "actor",
    "override_rationale",
    "payload_sha256",
    "prev_verdict",
];

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum DecisionRecordError {
    #[error("decision record refused: {0}")]
    Refused(String),
    #[error("decision record json: {0}")]
    Json(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecisionVerdict {
    Admit,
    Block,
    Override,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecisionThreatClass {
    None,
    Low,
    Medium,
    High,
    Critical,
    PayloadTooLarge,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecisionActor {
    Unattended,
    Human,
}

/// One replayable admission decision. Raw ingest is never a field.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecisionRecord {
    pub record_id: Uuid,
    pub timestamp_utc: DateTime<Utc>,
    pub policy_version: String,
    pub crate_version: String,
    pub surface: String,
    pub verdict: DecisionVerdict,
    pub threat_class: DecisionThreatClass,
    pub reason_codes: Vec<String>,
    pub actor: DecisionActor,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub override_rationale: Option<String>,
    pub payload_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prev_verdict: Option<DecisionVerdict>,
}

impl DecisionRecord {
    /// SHA-256 (lowercase hex) of UTF-8 ingest bytes. The only payload fingerprint.
    pub fn payload_sha256(content: &str) -> String {
        let digest = Sha256::digest(content.as_bytes());
        digest.iter().map(|b| format!("{b:02x}")).collect()
    }

    /// Record the living unattended gate. Calls `admit_or_block` as-is.
    /// Does not change thresholds. Does not store `content`.
    pub fn from_unattended_ingest(content: &str) -> Self {
        let payload_sha256 = Self::payload_sha256(content);
        match IngestionScanner::admit_or_block(content) {
            Ok(scan) => Self::from_scan(DecisionVerdict::Admit, &scan, payload_sha256),
            Err(MercySecurityError::PayloadTooLarge(_)) => Self::payload_too_large(payload_sha256),
            Err(_) => {
                let scan = IngestionScanner::scan_text(content);
                Self::from_scan(DecisionVerdict::Block, &scan, payload_sha256)
            }
        }
    }

    /// Human override of a prior record. Does **not** bypass the unattended scanner.
    /// Unattended Medium+ still blocks via `from_unattended_ingest` / `admit_or_block`.
    pub fn human_override(previous: &Self, rationale: &str) -> Result<Self, DecisionRecordError> {
        let rationale = rationale.trim();
        if rationale.is_empty() {
            return Err(DecisionRecordError::Refused(
                "override_rationale required for verdict=override".into(),
            ));
        }
        Ok(Self {
            record_id: Uuid::new_v4(),
            timestamp_utc: Utc::now(),
            policy_version: DECISION_RECORD_POLICY_VERSION.into(),
            crate_version: env!("CARGO_PKG_VERSION").into(),
            surface: DECISION_RECORD_SURFACE.into(),
            verdict: DecisionVerdict::Override,
            threat_class: previous.threat_class,
            reason_codes: vec!["human_override".into()],
            actor: DecisionActor::Human,
            override_rationale: Some(rationale.into()),
            payload_sha256: previous.payload_sha256.clone(),
            prev_verdict: Some(previous.verdict),
        })
    }

    pub fn to_log_json(&self) -> Result<String, DecisionRecordError> {
        let value = serde_json::to_value(self).map_err(|e| DecisionRecordError::Json(e.to_string()))?;
        refuse_raw_payload_object(&value)?;
        serde_json::to_string(&value).map_err(|e| DecisionRecordError::Json(e.to_string()))
    }

    pub fn from_log_json(s: &str) -> Result<Self, DecisionRecordError> {
        let value: serde_json::Value =
            serde_json::from_str(s).map_err(|e| DecisionRecordError::Json(e.to_string()))?;
        refuse_raw_payload_object(&value)?;
        let rec: Self =
            serde_json::from_value(value).map_err(|e| DecisionRecordError::Json(e.to_string()))?;
        rec.validate()?;
        Ok(rec)
    }

    fn from_scan(verdict: DecisionVerdict, scan: &IngestionScanResult, payload_sha256: String) -> Self {
        Self {
            record_id: Uuid::new_v4(),
            timestamp_utc: Utc::now(),
            policy_version: DECISION_RECORD_POLICY_VERSION.into(),
            crate_version: env!("CARGO_PKG_VERSION").into(),
            surface: DECISION_RECORD_SURFACE.into(),
            verdict,
            threat_class: threat_class_from_tier(scan.risk_tier),
            reason_codes: reason_codes_from_scan(scan),
            actor: DecisionActor::Unattended,
            override_rationale: None,
            payload_sha256,
            prev_verdict: None,
        }
    }

    fn payload_too_large(payload_sha256: String) -> Self {
        Self {
            record_id: Uuid::new_v4(),
            timestamp_utc: Utc::now(),
            policy_version: DECISION_RECORD_POLICY_VERSION.into(),
            crate_version: env!("CARGO_PKG_VERSION").into(),
            surface: DECISION_RECORD_SURFACE.into(),
            verdict: DecisionVerdict::Block,
            threat_class: DecisionThreatClass::PayloadTooLarge,
            reason_codes: vec![
                "payload_too_large".into(),
                format!("max_scan_bytes={}", crate::MAX_SCAN_BYTES),
            ],
            actor: DecisionActor::Unattended,
            override_rationale: None,
            payload_sha256,
            prev_verdict: None,
        }
    }

    fn validate(&self) -> Result<(), DecisionRecordError> {
        if self.surface != DECISION_RECORD_SURFACE {
            return Err(DecisionRecordError::Refused(format!(
                "surface must be {DECISION_RECORD_SURFACE}"
            )));
        }
        if self.payload_sha256.len() != 64
            || !self.payload_sha256.chars().all(|c| c.is_ascii_hexdigit())
        {
            return Err(DecisionRecordError::Refused(
                "payload_sha256 must be 64 lowercase/hex chars".into(),
            ));
        }
        if self.verdict == DecisionVerdict::Override {
            if self.actor != DecisionActor::Human {
                return Err(DecisionRecordError::Refused(
                    "override requires actor=human".into(),
                ));
            }
            if self
                .override_rationale
                .as_ref()
                .map(|s| s.trim().is_empty())
                .unwrap_or(true)
            {
                return Err(DecisionRecordError::Refused(
                    "override_rationale required for verdict=override".into(),
                ));
            }
            if self.prev_verdict.is_none() {
                return Err(DecisionRecordError::Refused(
                    "prev_verdict required for verdict=override".into(),
                ));
            }
        }
        Ok(())
    }
}

fn threat_class_from_tier(tier: RiskTier) -> DecisionThreatClass {
    match tier {
        RiskTier::None => DecisionThreatClass::None,
        RiskTier::Low => DecisionThreatClass::Low,
        RiskTier::Medium => DecisionThreatClass::Medium,
        RiskTier::High => DecisionThreatClass::High,
        RiskTier::Critical => DecisionThreatClass::Critical,
    }
}

fn threat_code(t: &IngestionThreat) -> &'static str {
    match t {
        IngestionThreat::RemoteCodeLoader => "remote_code_loader",
        IngestionThreat::TemplateInjection => "template_injection",
        IngestionThreat::SerializationGadget => "serialization_gadget",
        IngestionThreat::ShellProcessSpawn => "shell_process_spawn",
        IngestionThreat::NetworkCallback => "network_callback",
        IngestionThreat::ObfuscatedPayload => "obfuscated_payload",
        IngestionThreat::DatasetConfigInjection => "dataset_config_injection",
        IngestionThreat::CredentialHarvestPattern => "credential_harvest_pattern",
        IngestionThreat::CollusionPattern => "collusion_pattern",
        IngestionThreat::RewardHackingPattern => "reward_hacking_pattern",
        IngestionThreat::UnknownHighRisk => "unknown_high_risk",
    }
}

fn reason_codes_from_scan(scan: &IngestionScanResult) -> Vec<String> {
    let mut codes = vec![format!("tier:{}", scan.risk_tier.as_str())];
    for t in &scan.threats {
        let code = format!("threat:{}", threat_code(t));
        if !codes.contains(&code) {
            codes.push(code);
        }
    }
    for f in &scan.findings {
        let code = format!("signal:{}", f.signal);
        if !codes.contains(&code) {
            codes.push(code);
        }
    }
    codes
}

fn refuse_raw_payload_object(value: &serde_json::Value) -> Result<(), DecisionRecordError> {
    let obj = value.as_object().ok_or_else(|| {
        DecisionRecordError::Refused("decision record log must be a JSON object".into())
    })?;
    for key in obj.keys() {
        if FORBIDDEN_LOG_KEYS.iter().any(|k| k.eq_ignore_ascii_case(key)) {
            return Err(DecisionRecordError::Refused(format!(
                "log must not contain raw payload field '{key}'"
            )));
        }
        if !ALLOWED_LOG_KEYS.contains(&key.as_str()) {
            return Err(DecisionRecordError::Refused(format!(
                "unknown log field '{key}'"
            )));
        }
    }
    if !obj.contains_key("payload_sha256") {
        return Err(DecisionRecordError::Refused(
            "payload_sha256 required; raw payload is forbidden".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn payload_sha256_is_64_hex_and_stable() {
        let a = DecisionRecord::payload_sha256("tend the well");
        let b = DecisionRecord::payload_sha256("tend the well");
        assert_eq!(a, b);
        assert_eq!(a.len(), 64);
        assert!(a.chars().all(|c| matches!(c, '0'..='9' | 'a'..='f')));
        assert_ne!(a, DecisionRecord::payload_sha256("tend the well."));
    }

    #[test]
    fn empty_override_rationale_refused() {
        let prev = DecisionRecord::from_unattended_ingest("classroom notes. no remote loaders.");
        assert!(DecisionRecord::human_override(&prev, "   ").is_err());
    }
}
