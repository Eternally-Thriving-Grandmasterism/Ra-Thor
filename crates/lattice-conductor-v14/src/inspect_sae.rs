//! Mechanistic inspect adapter for wrapped-model calls (Opportunity 1).
//!
//! Emits a stable inspect packet from a pluggable SAE backend. The default
//! backend is a deterministic in-Rust dictionary stub. CI stays offline: no
//! weight download, no Python, no SAELens runtime.
//!
//! Steering proposals cannot apply unless they already passed
//! `wrap_model_output` / `handle_request`. An SAE packet cannot flip
//! Reject → Apply. Inspect-only mode works without steering.
//!
//! This is not METR. This is not sampler-weight constraint. Not a trained SAE.
//! Contact: info@Rathor.ai. Independent of xAI.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::evidence_chain::{hex_sha256, EvidenceDraft};
use crate::ra_thor_mercy_gated_api::{GateDecision, MercyApiResponse};

pub const CANONICAL_VERSION: &str = "RT-INSPECT-v1";
pub const STUB_BACKEND_ID: &str = "stub-dictionary-v0";
pub const WRAP_HOOK_SITE: &str = "wrap_model_output";
pub const WRAP_CIRCUIT_ID: &str = "conductor-v14-wrap";

/// Dictionary labels for the offline stub. Not a trained SAE catalog.
pub const STUB_FEATURE_IDS: [&str; 8] = [
    "feat.truth",
    "feat.order",
    "feat.love",
    "feat.compassion",
    "feat.service",
    "feat.abundance",
    "feat.joy",
    "feat.cosmic_harmony",
];

/// Inspect vs gated-steer. Default is inspect-only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum InspectMode {
    #[default]
    InspectOnly,
    SteerIfGated,
}

/// Gate face recorded on the packet. SAE cannot invent Allowed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InspectGateResult {
    InspectOnly,
    Allowed,
    Rejected { reason: String },
    BypassRejected,
}

impl InspectGateResult {
    pub fn as_label(&self) -> &'static str {
        match self {
            InspectGateResult::InspectOnly => "inspect_only",
            InspectGateResult::Allowed => "allowed",
            InspectGateResult::Rejected { .. } => "rejected",
            InspectGateResult::BypassRejected => "bypass_rejected",
        }
    }
}

/// Stable inspect packet. Hash excludes `gate_result` / `steering_applied`
/// so the feature body stays hash-stable across gate outcomes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InspectPacket {
    pub model_id: String,
    pub hook_site: String,
    pub feature_ids: Vec<String>,
    pub activations: Vec<f64>,
    pub proposed_steering_vector: Option<Vec<f64>>,
    pub gate_result: InspectGateResult,
    pub circuit_id: Option<String>,
    pub backend_id: String,
    pub packet_hash: String,
    pub steering_applied: bool,
}

impl InspectPacket {
    pub fn from_features(
        model_id: &str,
        hook_site: &str,
        features: SaeFeatures,
        proposed_steering_vector: Option<Vec<f64>>,
        circuit_id: Option<String>,
        backend_id: &str,
    ) -> Self {
        let mut packet = Self {
            model_id: model_id.to_string(),
            hook_site: hook_site.to_string(),
            feature_ids: features.feature_ids,
            activations: features.activations,
            proposed_steering_vector,
            gate_result: InspectGateResult::InspectOnly,
            circuit_id,
            backend_id: backend_id.to_string(),
            packet_hash: String::new(),
            steering_applied: false,
        };
        packet.packet_hash = packet.compute_hash();
        packet
    }

    pub fn canonical_preimage(&self) -> String {
        let mut map = BTreeMap::new();
        map.insert("activations", format_floats(&self.activations));
        map.insert(
            "circuit_id",
            self.circuit_id.clone().unwrap_or_default(),
        );
        map.insert("feature_ids", self.feature_ids.join(","));
        map.insert("hook_site", self.hook_site.clone());
        map.insert("model_id", self.model_id.clone());
        map.insert("backend_id", self.backend_id.clone());
        map.insert(
            "proposed_steering_vector",
            self.proposed_steering_vector
                .as_ref()
                .map(|v| format_floats(v))
                .unwrap_or_default(),
        );
        let body = serde_json::to_string(&map).unwrap_or_default();
        format!("{CANONICAL_VERSION}\n{body}")
    }

    pub fn compute_hash(&self) -> String {
        hex_sha256(self.canonical_preimage().as_bytes())
    }

    pub fn as_evidence_pointer(&self) -> String {
        format!("inspect-packet:{}", self.packet_hash)
    }

    /// Council-facing markdown dump. Not a live dashboard runtime.
    pub fn council_markdown(&self) -> String {
        let steer = match &self.proposed_steering_vector {
            Some(v) => format!("{} dims, applied={}", v.len(), self.steering_applied),
            None => "none (inspect-only)".into(),
        };
        format!(
            "## Inspect packet\n\n- model: `{}`\n- hook: `{}`\n- backend: `{}`\n- circuit: `{}`\n- features: {}\n- activations: {}\n- steering: {}\n- gate: {}\n- hash: `{}`\n",
            self.model_id,
            self.hook_site,
            self.backend_id,
            self.circuit_id.as_deref().unwrap_or("-"),
            self.feature_ids.join(", "),
            format_floats(&self.activations),
            steer,
            self.gate_result.as_label(),
            self.packet_hash
        )
    }

    pub fn council_json(&self) -> Result<String, SaeError> {
        serde_json::to_string_pretty(self).map_err(|e| SaeError::Serialize(e.to_string()))
    }
}

/// Sparse-ish feature readout from a backend. Not a trained catalog.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SaeFeatures {
    pub feature_ids: Vec<String>,
    pub activations: Vec<f64>,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum SaeError {
    #[error("inspect encode failed: {0}")]
    Encode(String),
    #[error("inspect serialize failed: {0}")]
    Serialize(String),
    #[error("steering that bypasses wrap_model_output / handle_request is Rejected")]
    BypassRejected,
    #[error("steering blocked by mercy gate: {0}")]
    GateRejected(String),
    #[error("SAELens / NNsight hook is design-only — not implemented")]
    NotImplemented,
    #[error("inspect lock poisoned — fail closed")]
    LockPoisoned,
}

/// Pluggable encode / optional steer-propose face.
pub trait SaeBackend {
    fn backend_id(&self) -> &'static str;
    fn encode(
        &self,
        model_id: &str,
        hook_site: &str,
        residual: &[f64],
    ) -> Result<SaeFeatures, SaeError>;
    fn propose_steering(&self, features: &SaeFeatures) -> Option<Vec<f64>>;
}

/// Deterministic in-Rust dictionary. No weights. Same bytes → same features.
#[derive(Debug, Clone, Default)]
pub struct DictionaryStubSae;

impl SaeBackend for DictionaryStubSae {
    fn backend_id(&self) -> &'static str {
        STUB_BACKEND_ID
    }

    fn encode(
        &self,
        _model_id: &str,
        _hook_site: &str,
        residual: &[f64],
    ) -> Result<SaeFeatures, SaeError> {
        Ok(SaeFeatures {
            feature_ids: STUB_FEATURE_IDS.iter().map(|s| (*s).to_string()).collect(),
            activations: dictionary_activations(residual),
        })
    }

    fn propose_steering(&self, features: &SaeFeatures) -> Option<Vec<f64>> {
        Some(features.activations.iter().map(|a| a * 0.1).collect())
    }
}

/// Tiny deterministic encoding of a residual (or text-derived bins) into 8 units.
pub fn dictionary_activations(residual: &[f64]) -> Vec<f64> {
    let n = STUB_FEATURE_IDS.len();
    let mut out = vec![0.0_f64; n];
    if residual.is_empty() {
        return out;
    }
    for (i, v) in residual.iter().enumerate() {
        if v.is_finite() {
            out[i % n] += v.abs();
        }
    }
    let max = out.iter().cloned().fold(0.0_f64, f64::max).max(1.0);
    for v in &mut out {
        *v /= max;
    }
    out
}

/// Derive a residual from model text. Offline. No live model.
pub fn residual_from_text(text: &str) -> Vec<f64> {
    let n = STUB_FEATURE_IDS.len();
    let mut sums = vec![0.0_f64; n];
    if text.is_empty() {
        return sums;
    }
    for (i, b) in text.as_bytes().iter().enumerate() {
        sums[i % n] += f64::from(*b);
    }
    let max = sums.iter().cloned().fold(0.0_f64, f64::max).max(1.0);
    for v in &mut sums {
        *v /= max;
    }
    sums
}

fn format_floats(values: &[f64]) -> String {
    values
        .iter()
        .map(|v| format!("{v:.6}"))
        .collect::<Vec<_>>()
        .join(",")
}

/// Encode one packet. Inspect-only leaves steering unset.
pub fn encode_inspect_packet(
    backend: &dyn SaeBackend,
    model_id: &str,
    hook_site: &str,
    residual: &[f64],
    circuit_id: Option<String>,
    mode: InspectMode,
) -> Result<InspectPacket, SaeError> {
    let features = backend.encode(model_id, hook_site, residual)?;
    let steering = match mode {
        InspectMode::InspectOnly => None,
        InspectMode::SteerIfGated => backend.propose_steering(&features),
    };
    Ok(InspectPacket::from_features(
        model_id,
        hook_site,
        features,
        steering,
        circuit_id,
        backend.backend_id(),
    ))
}

/// A wrap that never records an activation packet is not an inspectable apply.
pub fn is_inspectable_apply(accepted: bool, packet: Option<&InspectPacket>) -> bool {
    accepted && packet.is_some()
}

/// Steering follow-up. Missing wrap / handle receipt is BypassRejected.
/// A Rejected gate stays Rejected. SAE cannot flip it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SteeringProposal {
    pub packet_hash: String,
    pub vector: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SteeringApplyResult {
    pub applied: bool,
    pub gate_result: InspectGateResult,
    pub reason: String,
}

pub fn apply_steering_proposal(
    proposal: &SteeringProposal,
    wrap_receipt: Option<&MercyApiResponse>,
    handle_receipt: Option<&MercyApiResponse>,
) -> SteeringApplyResult {
    let _ = proposal;
    let Some(wrap) = wrap_receipt else {
        return SteeringApplyResult {
            applied: false,
            gate_result: InspectGateResult::BypassRejected,
            reason: SaeError::BypassRejected.to_string(),
        };
    };
    if !wrap.accepted {
        return reject_from_response(wrap, "wrap");
    }
    let Some(handle) = handle_receipt else {
        return SteeringApplyResult {
            applied: false,
            gate_result: InspectGateResult::BypassRejected,
            reason: SaeError::BypassRejected.to_string(),
        };
    };
    if !handle.accepted {
        return reject_from_response(handle, "handle_request");
    }
    SteeringApplyResult {
        applied: true,
        gate_result: InspectGateResult::Allowed,
        reason: "steering passed wrap_model_output and handle_request".into(),
    }
}

fn reject_from_response(resp: &MercyApiResponse, site: &str) -> SteeringApplyResult {
    let reason = match &resp.decision {
        GateDecision::Rejected { reason } => format!("{site}: {reason}"),
        GateDecision::Allowed => format!("{site}: not accepted"),
    };
    SteeringApplyResult {
        applied: false,
        gate_result: InspectGateResult::Rejected {
            reason: reason.clone(),
        },
        reason,
    }
}

/// Attach a packet pointer onto an evidence draft (Opportunity 3 types).
pub fn attach_packet_to_draft(draft: &mut EvidenceDraft, packet: &InspectPacket) {
    let pointer = packet.as_evidence_pointer();
    if !draft.evidence_pointers.iter().any(|p| p == &pointer) {
        draft.evidence_pointers.push(pointer);
    }
}

/// Recorder held by MercyGatedApi. Clone-safe. Default stub backend.
#[derive(Debug, Clone)]
pub struct InspectRecorder {
    pub mode: InspectMode,
    packets: Vec<InspectPacket>,
}

impl Default for InspectRecorder {
    fn default() -> Self {
        Self {
            mode: InspectMode::InspectOnly,
            packets: Vec::new(),
        }
    }
}

impl InspectRecorder {
    pub fn new(mode: InspectMode) -> Self {
        Self {
            mode,
            packets: Vec::new(),
        }
    }

    pub fn mode(&self) -> InspectMode {
        self.mode
    }

    pub fn set_mode(&mut self, mode: InspectMode) {
        self.mode = mode;
    }

    pub fn packets(&self) -> &[InspectPacket] {
        &self.packets
    }

    pub fn last(&self) -> Option<&InspectPacket> {
        self.packets.last()
    }

    pub fn record_text(
        &mut self,
        model_id: &str,
        hook_site: &str,
        text: &str,
        circuit_id: Option<String>,
    ) -> Result<InspectPacket, SaeError> {
        let residual = residual_from_text(text);
        let packet = encode_inspect_packet(
            &DictionaryStubSae,
            model_id,
            hook_site,
            &residual,
            circuit_id,
            self.mode,
        )?;
        self.packets.push(packet.clone());
        Ok(packet)
    }

    pub fn update_last_gate(&mut self, gate: InspectGateResult, steering_applied: bool) {
        if let Some(last) = self.packets.last_mut() {
            last.gate_result = gate;
            last.steering_applied = steering_applied;
        }
    }

    pub fn council_dashboard_markdown(&self) -> String {
        if self.packets.is_empty() {
            return "# Inspect dashboard\n\nNo packets. Inspect-only recorder is empty.\n".into();
        }
        let mut out = String::from("# Inspect dashboard\n\n");
        for packet in &self.packets {
            out.push_str(&packet.council_markdown());
            out.push('\n');
        }
        out
    }

    pub fn council_dashboard_json(&self) -> Result<String, SaeError> {
        serde_json::to_string_pretty(&self.packets).map_err(|e| SaeError::Serialize(e.to_string()))
    }
}

/// Design-only SAELens / HF / NNsight hook. Feature-flagged. No Python stack.
/// Does not download weights. Encode returns NotImplemented so CI stays on the stub.
#[cfg(feature = "sae-lens-hook")]
pub mod sae_lens_hook {
    use super::{SaeBackend, SaeError, SaeFeatures};

    /// Named encode/decode entries for a later SAELens-like adapter.
    #[derive(Debug, Clone)]
    pub struct HfNnsightHookSpec {
        pub model_id: String,
        pub hook_site: String,
        pub encode_entry: &'static str,
        pub decode_entry: &'static str,
    }

    impl Default for HfNnsightHookSpec {
        fn default() -> Self {
            Self {
                model_id: "unspecified".into(),
                hook_site: "residual_stream".into(),
                encode_entry: "saelens.encode",
                decode_entry: "saelens.decode",
            }
        }
    }

    impl SaeBackend for HfNnsightHookSpec {
        fn backend_id(&self) -> &'static str {
            "hf-nnsight-hook-spec"
        }

        fn encode(
            &self,
            _model_id: &str,
            _hook_site: &str,
            _residual: &[f64],
        ) -> Result<SaeFeatures, SaeError> {
            Err(SaeError::NotImplemented)
        }

        fn propose_steering(&self, _features: &SaeFeatures) -> Option<Vec<f64>> {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evidence_chain::{EvidenceChain, EvidenceDraft, EvidenceKind};
    use crate::ra_thor_mercy_gated_api::GateDecision;

    const FIXTURE_TEXT: &str = "Draft: tend the well and publish flow.";

    fn fixture_packet() -> InspectPacket {
        let residual = residual_from_text(FIXTURE_TEXT);
        encode_inspect_packet(
            &DictionaryStubSae,
            "grok-session",
            WRAP_HOOK_SITE,
            &residual,
            Some(WRAP_CIRCUIT_ID.into()),
            InspectMode::InspectOnly,
        )
        .unwrap()
    }

    #[test]
    fn packet_serializes() {
        let packet = fixture_packet();
        let json = packet.council_json().expect("serialize");
        let back: InspectPacket = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.model_id, packet.model_id);
        assert_eq!(back.hook_site, WRAP_HOOK_SITE);
        assert_eq!(back.feature_ids, packet.feature_ids);
        assert_eq!(back.packet_hash, packet.packet_hash);
        assert!(matches!(back.gate_result, InspectGateResult::InspectOnly));
        assert!(!back.steering_applied);
    }

    #[test]
    fn stub_sae_is_deterministic() {
        let a = fixture_packet();
        let b = fixture_packet();
        assert_eq!(a.activations, b.activations);
        assert_eq!(a.packet_hash, b.packet_hash);
        assert_eq!(a.packet_hash, a.compute_hash());
        assert_eq!(a.packet_hash.len(), 64);
        assert!(a.proposed_steering_vector.is_none());
    }

    #[test]
    fn fixture_hash_is_stable() {
        let raw = include_str!("../fixtures/inspect_packet_v0.json");
        let fx: serde_json::Value = serde_json::from_str(raw).expect("fixture json");
        let packet = fixture_packet();
        assert_eq!(packet.model_id, fx["model_id"]);
        assert_eq!(packet.hook_site, fx["hook_site"]);
        assert_eq!(packet.backend_id, fx["backend_id"]);
        assert_eq!(packet.circuit_id.as_deref(), fx["circuit_id"].as_str());
        assert_eq!(packet.packet_hash, fx["packet_hash"]);
        let expected_acts: Vec<f64> = fx["activations"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        for (got, want) in packet.activations.iter().zip(expected_acts.iter()) {
            assert!((got - want).abs() < 1e-9);
        }
    }

    #[test]
    fn wrap_without_packet_is_not_inspectable_apply() {
        assert!(!is_inspectable_apply(true, None));
        let packet = fixture_packet();
        assert!(is_inspectable_apply(true, Some(&packet)));
        assert!(!is_inspectable_apply(false, Some(&packet)));
    }

    #[test]
    fn steering_bypass_without_wrap_is_rejected() {
        let proposal = SteeringProposal {
            packet_hash: "abc".into(),
            vector: vec![0.1, 0.0],
        };
        let miss = apply_steering_proposal(&proposal, None, None);
        assert!(!miss.applied);
        assert_eq!(miss.gate_result, InspectGateResult::BypassRejected);
    }

    #[test]
    fn steering_path_blocked_when_gate_rejects() {
        let proposal = SteeringProposal {
            packet_hash: fixture_packet().packet_hash,
            vector: vec![0.1],
        };
        let wrap = MercyApiResponse {
            accepted: false,
            decision: GateDecision::Rejected {
                reason: "Layer 0 ingest".into(),
            },
            message: "Rejected".into(),
            mercy_score: 0.99,
            gates_checked: vec![],
            timestamp: 1,
            cosmic_loop_ready: true,
            evidence_hash: None,
            inspect_packet_hash: None,
            wrap_account: crate::ra_thor_mercy_gated_api::WrapAccount::default(),
        };
        let handle = wrap.clone();
        let result = apply_steering_proposal(&proposal, Some(&wrap), Some(&handle));
        assert!(!result.applied);
        assert!(matches!(
            result.gate_result,
            InspectGateResult::Rejected { .. }
        ));
    }

    #[test]
    fn packet_attaches_to_evidence_record() {
        let packet = fixture_packet();
        let mut draft = EvidenceDraft {
            subject: "wrap:operator".into(),
            input: "payload".into(),
            claim: "wrap-apply".into(),
            evidence_pointers: vec!["layer0".into()],
            directive: "accept".into(),
            scope: "lattice-conductor-v14".into(),
            kind: EvidenceKind::Wrap,
            decision: "accept".into(),
            timestamp: 1,
            actor: "operator".into(),
            auditor: "evidence-face".into(),
        };
        attach_packet_to_draft(&mut draft, &packet);
        assert!(draft
            .evidence_pointers
            .iter()
            .any(|p| p == &packet.as_evidence_pointer()));

        let mut chain = EvidenceChain::new();
        let rec = chain.append(draft).expect("append wrap+inspect");
        assert!(rec
            .evidence_pointers
            .iter()
            .any(|p| p.starts_with("inspect-packet:")));
        chain.gate_side_effect(Some(&rec.hash)).unwrap();
    }

    #[test]
    fn sae_cannot_flip_reject_to_apply() {
        let wrap = MercyApiResponse {
            accepted: false,
            decision: GateDecision::Rejected {
                reason: "Layer 0".into(),
            },
            message: "Rejected".into(),
            mercy_score: 0.99,
            gates_checked: vec![],
            timestamp: 1,
            cosmic_loop_ready: true,
            evidence_hash: None,
            inspect_packet_hash: None,
            wrap_account: crate::ra_thor_mercy_gated_api::WrapAccount::default(),
        };
        let mut packet = fixture_packet();
        packet.gate_result = InspectGateResult::Allowed;
        let mock_council_approve = true;
        assert!(mock_council_approve);
        let result = apply_steering_proposal(
            &SteeringProposal {
                packet_hash: packet.packet_hash.clone(),
                vector: vec![1.0],
            },
            Some(&wrap),
            Some(&wrap),
        );
        assert!(!result.applied);
        assert!(!wrap.accepted);
        assert!(!is_inspectable_apply(wrap.accepted, Some(&packet)));
    }

    #[test]
    fn inspect_only_does_not_propose_steering() {
        let mut rec = InspectRecorder::new(InspectMode::InspectOnly);
        let packet = rec
            .record_text("grok-session", WRAP_HOOK_SITE, FIXTURE_TEXT, None)
            .unwrap();
        assert!(packet.proposed_steering_vector.is_none());
        assert!(!packet.steering_applied);
        assert!(rec.council_dashboard_markdown().contains("Inspect packet"));
    }

    #[cfg(feature = "sae-lens-hook")]
    #[test]
    fn sae_lens_hook_is_design_only() {
        use crate::inspect_sae::sae_lens_hook::HfNnsightHookSpec;
        let hook = HfNnsightHookSpec::default();
        assert_eq!(hook.encode_entry, "saelens.encode");
        assert!(matches!(
            hook.encode("x", "residual_stream", &[0.0]),
            Err(SaeError::NotImplemented)
        ));
    }

    #[test]
    fn steer_mode_proposes_deterministic_vector() {
        let mut rec = InspectRecorder::new(InspectMode::SteerIfGated);
        let a = rec
            .record_text("grok-session", WRAP_HOOK_SITE, FIXTURE_TEXT, None)
            .unwrap();
        let b = rec
            .record_text("grok-session", WRAP_HOOK_SITE, FIXTURE_TEXT, None)
            .unwrap();
        assert_eq!(a.proposed_steering_vector, b.proposed_steering_vector);
        let steer = a.proposed_steering_vector.expect("steer mode proposes");
        assert_eq!(steer.len(), STUB_FEATURE_IDS.len());
        assert!(!a.steering_applied);
    }
}
