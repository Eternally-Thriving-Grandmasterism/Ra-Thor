//! Ra-Thor-native runtime evidence chain (AIREP-inspired, not AIREP-compatible).
//!
//! One hash-chained record per verifier decision and per gated submission.
//! The evidential face is a gate: missing record or broken chain → no apply.
//! Silent prompt / harness mutation is forbidden. Council 13 / human override
//! remains. A council cannot delete the chain to hide a Reject.
//!
//! Contact: info@Rathor.ai. Independent of xAI. Not a signed production ledger.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

pub const GENESIS_HASH: &str =
    "0000000000000000000000000000000000000000000000000000000000000000";
pub const CANONICAL_VERSION: &str = "RT-EVIDENCE-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceKind {
    Council,
    Wrap,
    Tool,
    Lipschitz,
    SelfEvolution,
    Inspect,
    Prefix,
}

impl EvidenceKind {
    pub fn as_str(self) -> &'static str {
        match self {
            EvidenceKind::Council => "council",
            EvidenceKind::Wrap => "wrap",
            EvidenceKind::Tool => "tool",
            EvidenceKind::Lipschitz => "lipschitz",
            EvidenceKind::SelfEvolution => "self_evolution",
            EvidenceKind::Inspect => "inspect",
            EvidenceKind::Prefix => "prefix",
        }
    }
}

/// Draft fields before hash / previous-hash are sealed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceDraft {
    pub subject: String,
    pub input: String,
    pub claim: String,
    pub evidence_pointers: Vec<String>,
    pub directive: String,
    pub scope: String,
    pub kind: EvidenceKind,
    pub decision: String,
    pub timestamp: u64,
    pub actor: String,
    pub auditor: String,
}

/// Offline-checkable evidence row.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceRecord {
    pub subject: String,
    pub input: String,
    pub claim: String,
    pub evidence_pointers: Vec<String>,
    pub directive: String,
    pub scope: String,
    pub kind: EvidenceKind,
    pub decision: String,
    pub timestamp: u64,
    pub actor: String,
    pub auditor: String,
    pub previous_hash: String,
    pub hash: String,
}

impl EvidenceRecord {
    pub fn from_draft(draft: EvidenceDraft, previous_hash: String) -> Self {
        let mut rec = Self {
            subject: sanitize(&draft.subject),
            input: sanitize(&draft.input),
            claim: sanitize(&draft.claim),
            evidence_pointers: draft
                .evidence_pointers
                .into_iter()
                .map(|p| sanitize(&p))
                .collect(),
            directive: sanitize(&draft.directive),
            scope: sanitize(&draft.scope),
            kind: draft.kind,
            decision: sanitize(&draft.decision),
            timestamp: draft.timestamp,
            actor: sanitize(&draft.actor),
            auditor: sanitize(&draft.auditor),
            previous_hash,
            hash: String::new(),
        };
        rec.hash = rec.compute_hash();
        rec
    }

    pub fn canonical_preimage(&self) -> String {
        // Sorted keys via BTreeMap so a later agent can re-hash without serde field order.
        let mut map = BTreeMap::new();
        map.insert("actor", self.actor.clone());
        map.insert("auditor", self.auditor.clone());
        map.insert("claim", self.claim.clone());
        map.insert("decision", self.decision.clone());
        map.insert("directive", self.directive.clone());
        map.insert(
            "evidence_pointers",
            self.evidence_pointers.join(","),
        );
        map.insert("input", self.input.clone());
        map.insert("kind", self.kind.as_str().to_string());
        map.insert("previous_hash", self.previous_hash.clone());
        map.insert("scope", self.scope.clone());
        map.insert("subject", self.subject.clone());
        map.insert("timestamp", self.timestamp.to_string());
        let body = serde_json::to_string(&map).unwrap_or_default();
        format!("{CANONICAL_VERSION}\n{body}")
    }

    pub fn compute_hash(&self) -> String {
        hex_sha256(self.canonical_preimage().as_bytes())
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum EvidenceError {
    #[error("missing evidence record — no apply")]
    MissingRecord,
    #[error("broken evidence chain at index {index}: {reason}")]
    BrokenChain { index: usize, reason: String },
    #[error("council self-audit forbidden (actor == auditor = {actor})")]
    SelfAudit { actor: String },
    #[error("erase forbidden — council cannot delete the chain to hide a Reject")]
    EraseForbidden,
    #[error("evidence lock poisoned — fail closed")]
    LockPoisoned,
}

#[derive(Debug, Default, Clone)]
pub struct EvidenceChain {
    records: Vec<EvidenceRecord>,
}

impl EvidenceChain {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn records(&self) -> &[EvidenceRecord] {
        &self.records
    }

    pub fn last(&self) -> Option<&EvidenceRecord> {
        self.records.last()
    }

    pub fn last_hash(&self) -> String {
        self.records
            .last()
            .map(|r| r.hash.clone())
            .unwrap_or_else(|| GENESIS_HASH.to_string())
    }

    pub fn append(&mut self, draft: EvidenceDraft) -> Result<EvidenceRecord, EvidenceError> {
        if draft.kind == EvidenceKind::Council && draft.actor == draft.auditor {
            return Err(EvidenceError::SelfAudit {
                actor: draft.actor,
            });
        }
        self.verify()?;
        let previous_hash = self.last_hash();
        let rec = EvidenceRecord::from_draft(draft, previous_hash);
        self.records.push(rec.clone());
        Ok(rec)
    }

    pub fn verify(&self) -> Result<(), EvidenceError> {
        Self::verify_offline(&self.records)
    }

    /// Recompute the hash chain with no network.
    pub fn verify_offline(records: &[EvidenceRecord]) -> Result<(), EvidenceError> {
        let mut prev = GENESIS_HASH.to_string();
        for (index, rec) in records.iter().enumerate() {
            if rec.kind == EvidenceKind::Council && rec.actor == rec.auditor {
                return Err(EvidenceError::SelfAudit {
                    actor: rec.actor.clone(),
                });
            }
            if rec.previous_hash != prev {
                return Err(EvidenceError::BrokenChain {
                    index,
                    reason: format!(
                        "previous_hash {} != expected {}",
                        rec.previous_hash, prev
                    ),
                });
            }
            let recomputed = rec.compute_hash();
            if recomputed != rec.hash {
                return Err(EvidenceError::BrokenChain {
                    index,
                    reason: format!("hash {} != recomputed {}", rec.hash, recomputed),
                });
            }
            prev = rec.hash.clone();
        }
        Ok(())
    }

    /// Side effect gate: apply is forbidden without a live, chained record.
    pub fn gate_side_effect(&self, record_hash: Option<&str>) -> Result<(), EvidenceError> {
        let Some(h) = record_hash.map(str::trim).filter(|s| !s.is_empty()) else {
            return Err(EvidenceError::MissingRecord);
        };
        self.verify()?;
        if !self.records.iter().any(|r| r.hash == h) {
            return Err(EvidenceError::MissingRecord);
        }
        Ok(())
    }

    pub fn erase_record(&mut self, _hash: &str) -> Result<(), EvidenceError> {
        Err(EvidenceError::EraseForbidden)
    }

    /// Test helper: break the chain without providing a public delete.
    #[cfg(test)]
    pub fn tamper_last_previous_hash(&mut self) {
        if let Some(last) = self.records.last_mut() {
            last.previous_hash = "deadbeef".repeat(8);
        }
    }
}

pub fn hex_sha256(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

pub fn payload_digest(payload: &str) -> String {
    hex_sha256(payload.as_bytes())
}

fn sanitize(s: &str) -> String {
    s.replace(['\n', '\r'], " ")
}

/// Convert an Allowed-looking apply into a Reject when the evidential face fails.
pub fn apply_without_record_is_not_apply(has_record: bool, chain_ok: bool) -> bool {
    has_record && chain_ok
}

#[cfg(test)]
mod tests {
    use super::*;

    fn draft(kind: EvidenceKind, actor: &str, auditor: &str, ts: u64, claim: &str) -> EvidenceDraft {
        EvidenceDraft {
            subject: format!("{}:{actor}", kind.as_str()),
            input: payload_digest(claim),
            claim: claim.into(),
            evidence_pointers: vec!["layer0".into(), "fixture".into()],
            directive: "accept".into(),
            scope: "lattice-conductor-v14".into(),
            kind,
            decision: "accept".into(),
            timestamp: ts,
            actor: actor.into(),
            auditor: auditor.into(),
        }
    }

    #[test]
    fn three_row_fixture_recomputes_hash_chain() {
        let raw = include_str!("../fixtures/evidence_chain_three_row_v0.json");
        let rows: Vec<EvidenceDraft> = serde_json::from_str(raw).expect("three-row fixture");
        assert_eq!(rows.len(), 3);

        let mut chain = EvidenceChain::new();
        for row in rows {
            chain.append(row).expect("append fixture row");
        }
        EvidenceChain::verify_offline(chain.records()).expect("recomputed chain");
        assert_eq!(chain.records()[0].previous_hash, GENESIS_HASH);
        assert_eq!(
            chain.records()[1].previous_hash,
            chain.records()[0].hash
        );
        assert_eq!(
            chain.records()[2].previous_hash,
            chain.records()[1].hash
        );
        for rec in chain.records() {
            assert_eq!(rec.hash, rec.compute_hash());
            assert_eq!(rec.hash.len(), 64);
        }
    }

    #[test]
    fn missing_record_fails_side_effect() {
        let chain = EvidenceChain::new();
        assert_eq!(
            chain.gate_side_effect(None),
            Err(EvidenceError::MissingRecord)
        );
        assert!(!apply_without_record_is_not_apply(false, true));
    }

    #[test]
    fn broken_previous_hash_is_rejected() {
        let mut chain = EvidenceChain::new();
        chain
            .append(draft(EvidenceKind::Wrap, "operator", "evidence-face", 1, "a"))
            .unwrap();
        chain
            .append(draft(EvidenceKind::Tool, "operator", "evidence-face", 2, "b"))
            .unwrap();
        chain.tamper_last_previous_hash();
        assert!(matches!(
            chain.verify(),
            Err(EvidenceError::BrokenChain { index: 1, .. })
        ));
        let hash = chain.last().unwrap().hash.clone();
        assert!(chain.gate_side_effect(Some(&hash)).is_err());
    }

    #[test]
    fn council_self_audit_is_rejected() {
        let mut chain = EvidenceChain::new();
        let err = chain
            .append(draft(
                EvidenceKind::Council,
                "council-13",
                "council-13",
                3,
                "score own homework",
            ))
            .unwrap_err();
        assert!(matches!(err, EvidenceError::SelfAudit { .. }));
        assert!(chain.records().is_empty());
    }

    #[test]
    fn erase_is_forbidden() {
        let mut chain = EvidenceChain::new();
        let rec = chain
            .append(draft(EvidenceKind::Wrap, "op", "evidence-face", 1, "x"))
            .unwrap();
        assert_eq!(
            chain.erase_record(&rec.hash),
            Err(EvidenceError::EraseForbidden)
        );
        assert_eq!(chain.records().len(), 1);
    }

    #[test]
    fn side_effect_passes_only_with_live_hash() {
        let mut chain = EvidenceChain::new();
        let rec = chain
            .append(draft(EvidenceKind::Tool, "op", "evidence-face", 1, "tool"))
            .unwrap();
        chain.gate_side_effect(Some(&rec.hash)).unwrap();
        assert!(apply_without_record_is_not_apply(true, true));
    }
}
