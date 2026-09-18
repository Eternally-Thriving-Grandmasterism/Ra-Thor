//! Prefix / pipeline risk for multi-step Lattice Conductor paths (Opportunity 4).
//!
//! Rule-based and deterministic. Scores every prefix, not only the final string.
//! High risk halts or escalates to council before later steps apply.
//! Harness evolution is a gated proposal object only — never applied silently.
//!
//! Observer-model hook is behind `PrefixConfig.observer_model` (default off).
//! Default CI does not call a second LLM.
//!
//! Not METR. Not a valence number. Independent of xAI.
//! Contact: info@Rathor.ai

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::evidence_chain::{
    hex_sha256, payload_digest, EvidenceChain, EvidenceDraft, EvidenceError, EvidenceKind,
};
use crate::patsagi_governance::{PatsagiCouncilSimulator, PatsagiReviewRequest};
use crate::ra_thor_mercy_gated_api::{ApiRequestKind, MercyApiRequest, MercyGatedApi};
use crate::wrap_model_output::{wrap_model_output, ModelSurface};
use crate::CouncilArbitrationEngine;

pub const CANONICAL_VERSION: &str = "RT-PREFIX-v1";
pub const DEFAULT_THRESHOLD: u32 = 50;
/// Claimed-mercy fail-closed floor for prefix scoring. Not the valence floor.
pub const MIN_CLAIMED_MERCY: f64 = 0.75;
pub const COUNCIL_13: u32 = 13;
pub const WEIGHT_LOOP: u32 = 60;
pub const WEIGHT_SKIPPED_VERIFY: u32 = 55;
pub const WEIGHT_TOOL_BEFORE_PLAN: u32 = 55;
pub const WEIGHT_REPEATED_TOOL: u32 = 55;

/// Canonical trajectory step taxonomy matching lived Ra-Thor events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrajectoryStepKind {
    /// Safe-read / tree walk (`get_tree_safe`, search).
    Search,
    /// File or health read (`get_file_contents_safe`, HealthCheck).
    Read,
    /// Apply-class payload mutation.
    Edit,
    /// Tool fire (`EvidenceKind::Tool`, queued intent).
    Tool,
    /// Council / PATSAGi vote.
    Vote,
    /// `wrap_model_output`.
    Wrap,
    /// Lipschitz / evidence / inspect verify.
    Verify,
    /// Halt, reject, or human override.
    Revert,
}

impl TrajectoryStepKind {
    pub fn as_str(self) -> &'static str {
        match self {
            TrajectoryStepKind::Search => "search",
            TrajectoryStepKind::Read => "read",
            TrajectoryStepKind::Edit => "edit",
            TrajectoryStepKind::Tool => "tool",
            TrajectoryStepKind::Vote => "vote",
            TrajectoryStepKind::Wrap => "wrap",
            TrajectoryStepKind::Verify => "verify",
            TrajectoryStepKind::Revert => "revert",
        }
    }

    /// Apply-class kinds that must be followed by Verify before another apply.
    pub fn needs_verify(self) -> bool {
        matches!(
            self,
            TrajectoryStepKind::Edit | TrajectoryStepKind::Tool | TrajectoryStepKind::Wrap
        )
    }

    pub fn is_plan(self) -> bool {
        matches!(self, TrajectoryStepKind::Search | TrajectoryStepKind::Read)
    }

    pub fn is_toolish(self) -> bool {
        matches!(self, TrajectoryStepKind::Tool | TrajectoryStepKind::Edit)
    }
}

/// TRACES-like automaton states for a prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrefixState {
    Idle,
    Planned,
    Acting,
    Verifying,
    Complete,
    Halted,
    Escalated,
    Reverted,
}

impl PrefixState {
    pub fn as_str(self) -> &'static str {
        match self {
            PrefixState::Idle => "idle",
            PrefixState::Planned => "planned",
            PrefixState::Acting => "acting",
            PrefixState::Verifying => "verifying",
            PrefixState::Complete => "complete",
            PrefixState::Halted => "halted",
            PrefixState::Escalated => "escalated",
            PrefixState::Reverted => "reverted",
        }
    }

    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            PrefixState::Halted | PrefixState::Escalated | PrefixState::Reverted | PrefixState::Complete
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryStep {
    pub kind: TrajectoryStepKind,
    pub payload: String,
    pub claimed_mercy: f64,
    pub actor: String,
    #[serde(default)]
    pub tool_id: Option<String>,
}

impl TrajectoryStep {
    pub fn new(kind: TrajectoryStepKind, payload: impl Into<String>, claimed_mercy: f64) -> Self {
        Self {
            kind,
            payload: payload.into(),
            claimed_mercy,
            actor: "operator".into(),
            tool_id: None,
        }
    }

    pub fn with_tool_id(mut self, id: impl Into<String>) -> Self {
        self.tool_id = Some(id.into());
        self
    }

    pub fn with_actor(mut self, actor: impl Into<String>) -> Self {
        self.actor = actor.into();
        self
    }

    pub fn identity_key(&self) -> String {
        format!(
            "{}:{}:{}",
            self.kind.as_str(),
            self.tool_id.as_deref().unwrap_or("-"),
            payload_digest(&self.payload)
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrefixFinding {
    Loop { cycle_len: usize, repeats: usize },
    SkippedVerify { previous: String, current: String },
    ToolBeforePlan { kind: String },
    RepeatedIdenticalTool { tool_key: String, count: usize },
    LowClaimedMercy { claimed_millis: i64, threshold_millis: i64 },
    Observer { extra: u32 },
}

impl PrefixFinding {
    pub fn weight(&self) -> u32 {
        match self {
            PrefixFinding::Loop { .. } => WEIGHT_LOOP,
            PrefixFinding::SkippedVerify { .. } => WEIGHT_SKIPPED_VERIFY,
            PrefixFinding::ToolBeforePlan { .. } => WEIGHT_TOOL_BEFORE_PLAN,
            PrefixFinding::RepeatedIdenticalTool { .. } => WEIGHT_REPEATED_TOOL,
            PrefixFinding::LowClaimedMercy { .. } => WEIGHT_SKIPPED_VERIFY,
            PrefixFinding::Observer { extra } => *extra,
        }
    }

    pub fn as_label(&self) -> &'static str {
        match self {
            PrefixFinding::Loop { .. } => "loop",
            PrefixFinding::SkippedVerify { .. } => "skipped_verify",
            PrefixFinding::ToolBeforePlan { .. } => "tool_before_plan",
            PrefixFinding::RepeatedIdenticalTool { .. } => "repeated_identical_tool",
            PrefixFinding::LowClaimedMercy { .. } => "low_claimed_mercy",
            PrefixFinding::Observer { .. } => "observer",
        }
    }

    pub fn blocks_live_apply(&self) -> bool {
        matches!(
            self,
            PrefixFinding::Loop { .. }
                | PrefixFinding::SkippedVerify { .. }
                | PrefixFinding::ToolBeforePlan { .. }
                | PrefixFinding::RepeatedIdenticalTool { .. }
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrefixAction {
    Continue,
    Halt { reason: String },
    Escalate { reason: String, councils: Vec<u32> },
}

impl PrefixAction {
    pub fn as_kind(&self) -> &'static str {
        match self {
            PrefixAction::Continue => "continue",
            PrefixAction::Halt { .. } => "halt",
            PrefixAction::Escalate { .. } => "escalate",
        }
    }

    pub fn is_stop(&self) -> bool {
        !matches!(self, PrefixAction::Continue)
    }
}

/// TraceProbe-style diagnostics for one prefix.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceProbe {
    pub risk: u32,
    pub state: PrefixState,
    pub findings: Vec<PrefixFinding>,
    pub action: PrefixAction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrefixStepRecord {
    pub index: usize,
    pub kind: TrajectoryStepKind,
    pub payload_digest: String,
    pub claimed_mercy: f64,
    pub risk_after: u32,
    pub findings: Vec<PrefixFinding>,
    pub state: PrefixState,
    pub action: PrefixAction,
    pub layer0_accepted: Option<bool>,
    #[serde(default)]
    pub evidence_hash: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrefixSnapshot {
    pub risk: u32,
    pub findings: Vec<PrefixFinding>,
    pub state: PrefixState,
    pub action: PrefixAction,
    pub records: Vec<PrefixStepRecord>,
}

impl PrefixSnapshot {
    pub fn probe(&self) -> TraceProbe {
        TraceProbe {
            risk: self.risk,
            state: self.state,
            findings: self.findings.clone(),
            action: self.action.clone(),
        }
    }

    pub fn canonical_preimage(&self) -> String {
        let mut map = BTreeMap::new();
        map.insert("risk", self.risk.to_string());
        map.insert("state", self.state.as_str().to_string());
        map.insert("action", self.action.as_kind().to_string());
        let steps: Vec<String> = self
            .records
            .iter()
            .map(|r| {
                format!(
                    "{}:{}:{}:{}",
                    r.index,
                    r.kind.as_str(),
                    r.risk_after,
                    r.action.as_kind()
                )
            })
            .collect();
        map.insert("steps", steps.join("|"));
        let findings: Vec<String> = self
            .findings
            .iter()
            .map(|f| f.as_label().to_string())
            .collect();
        map.insert("findings", findings.join(","));
        let body = serde_json::to_string(&map).unwrap_or_default();
        format!("{CANONICAL_VERSION}\n{body}")
    }

    pub fn compute_hash(&self) -> String {
        hex_sha256(self.canonical_preimage().as_bytes())
    }
}

#[derive(Debug, Clone)]
pub struct PrefixConfig {
    pub risk_threshold: u32,
    pub min_claimed_mercy: f64,
    /// Observer-model hook. Default off. Never required in CI.
    pub observer_model: bool,
}

impl Default for PrefixConfig {
    fn default() -> Self {
        Self {
            risk_threshold: DEFAULT_THRESHOLD,
            min_claimed_mercy: MIN_CLAIMED_MERCY,
            observer_model: false,
        }
    }
}

/// Optional extra risk. Default CI never enables this. Stub is not an LLM.
pub trait PrefixObserver {
    fn extra_risk(&self, steps: &[TrajectoryStep]) -> u32;
}

pub struct NullObserver;

impl PrefixObserver for NullObserver {
    fn extra_risk(&self, _steps: &[TrajectoryStep]) -> u32 {
        0
    }
}

impl PrefixConfig {
    pub fn observer_bonus(&self, steps: &[TrajectoryStep]) -> u32 {
        if !self.observer_model {
            return 0;
        }
        NullObserver.extra_risk(steps)
    }
}

fn mercy_millis(v: f64) -> i64 {
    if !v.is_finite() {
        return i64::MIN;
    }
    (v * 1000.0).round() as i64
}

fn detect_loop(kinds: &[TrajectoryStepKind]) -> Option<PrefixFinding> {
    let n = kinds.len();
    if n >= 3 {
        let a = kinds[n - 1];
        if kinds[n - 2] == a && kinds[n - 3] == a {
            return Some(PrefixFinding::Loop {
                cycle_len: 1,
                repeats: 3,
            });
        }
    }
    if n >= 4 {
        let a = kinds[n - 4];
        let b = kinds[n - 3];
        if a != b && kinds[n - 2] == a && kinds[n - 1] == b {
            return Some(PrefixFinding::Loop {
                cycle_len: 2,
                repeats: 2,
            });
        }
    }
    if n >= 6 {
        let a = kinds[n - 6];
        let b = kinds[n - 5];
        let c = kinds[n - 4];
        if kinds[n - 3] == a && kinds[n - 2] == b && kinds[n - 1] == c {
            return Some(PrefixFinding::Loop {
                cycle_len: 3,
                repeats: 2,
            });
        }
    }
    None
}

fn transition(state: PrefixState, kind: TrajectoryStepKind) -> PrefixState {
    if matches!(
        state,
        PrefixState::Halted | PrefixState::Escalated | PrefixState::Reverted
    ) {
        return state;
    }
    match kind {
        TrajectoryStepKind::Revert => PrefixState::Reverted,
        TrajectoryStepKind::Search | TrajectoryStepKind::Read => PrefixState::Planned,
        TrajectoryStepKind::Edit | TrajectoryStepKind::Tool | TrajectoryStepKind::Wrap => {
            PrefixState::Acting
        }
        TrajectoryStepKind::Verify => {
            if matches!(state, PrefixState::Acting | PrefixState::Verifying) {
                if state == PrefixState::Verifying {
                    PrefixState::Complete
                } else {
                    PrefixState::Verifying
                }
            } else if state == PrefixState::Planned {
                PrefixState::Complete
            } else {
                PrefixState::Verifying
            }
        }
        TrajectoryStepKind::Vote => state,
    }
}

/// Score one prefix. Pure. No network. No second model.
pub fn score_prefix(steps: &[TrajectoryStep], config: &PrefixConfig) -> PrefixSnapshot {
    let mut records = Vec::with_capacity(steps.len());
    let mut findings_now: Vec<PrefixFinding> = Vec::new();
    let mut state = PrefixState::Idle;
    let mut risk = 0u32;
    let mut action = PrefixAction::Continue;
    let mut saw_plan = false;
    let mut pending_verify: Option<TrajectoryStepKind> = None;
    let mut last_tool_key: Option<String> = None;
    let mut last_tool_count = 0usize;
    let mut kinds: Vec<TrajectoryStepKind> = Vec::new();

    for (index, step) in steps.iter().enumerate() {
        kinds.push(step.kind);
        if step.kind.is_plan() {
            saw_plan = true;
        }

        let mut step_findings: Vec<PrefixFinding> = Vec::new();

        if !step.claimed_mercy.is_finite()
            || step.claimed_mercy < config.min_claimed_mercy
        {
            step_findings.push(PrefixFinding::LowClaimedMercy {
                claimed_millis: mercy_millis(step.claimed_mercy),
                threshold_millis: mercy_millis(config.min_claimed_mercy),
            });
        }

        if step.kind.is_toolish() && !saw_plan {
            step_findings.push(PrefixFinding::ToolBeforePlan {
                kind: step.kind.as_str().into(),
            });
        }

        if step.kind.needs_verify() {
            if let Some(prev) = pending_verify {
                step_findings.push(PrefixFinding::SkippedVerify {
                    previous: prev.as_str().into(),
                    current: step.kind.as_str().into(),
                });
            }
            pending_verify = Some(step.kind);
        } else if step.kind == TrajectoryStepKind::Verify {
            pending_verify = None;
        }

        if step.kind == TrajectoryStepKind::Tool {
            let key = step.identity_key();
            if last_tool_key.as_deref() == Some(key.as_str()) {
                last_tool_count += 1;
                step_findings.push(PrefixFinding::RepeatedIdenticalTool {
                    tool_key: key,
                    count: last_tool_count,
                });
            } else {
                last_tool_key = Some(key);
                last_tool_count = 1;
            }
        } else {
            last_tool_key = None;
            last_tool_count = 0;
        }

        if let Some(loop_f) = detect_loop(&kinds) {
            step_findings.push(loop_f);
        }

        if config.observer_model {
            let extra = config.observer_bonus(&steps[..=index]);
            if extra > 0 {
                step_findings.push(PrefixFinding::Observer { extra });
            }
        }

        risk = step_findings.iter().map(|f| f.weight()).sum::<u32>().min(100);
        findings_now = step_findings.clone();
        action = action_from_findings(risk, config.risk_threshold, &findings_now);
        state = if let PrefixAction::Halt { .. } = &action {
            PrefixState::Halted
        } else if let PrefixAction::Escalate { .. } = &action {
            PrefixState::Escalated
        } else if step.kind == TrajectoryStepKind::Revert {
            PrefixState::Reverted
        } else {
            let next = transition(state, step.kind);
            if step.kind == TrajectoryStepKind::Verify && pending_verify.is_none() {
                PrefixState::Complete
            } else {
                next
            }
        };

        records.push(PrefixStepRecord {
            index,
            kind: step.kind,
            payload_digest: payload_digest(&step.payload),
            claimed_mercy: step.claimed_mercy,
            risk_after: risk,
            findings: step_findings,
            state,
            action: action.clone(),
            layer0_accepted: None,
            evidence_hash: None,
        });

        if action.is_stop() {
            break;
        }
    }

    PrefixSnapshot {
        risk,
        findings: findings_now,
        state,
        action,
        records,
    }
}

fn action_from_findings(risk: u32, threshold: u32, findings: &[PrefixFinding]) -> PrefixAction {
    if risk < threshold {
        return PrefixAction::Continue;
    }
    let has_loop = findings
        .iter()
        .any(|f| matches!(f, PrefixFinding::Loop { .. }));
    let reason = findings
        .iter()
        .map(|f| f.as_label())
        .collect::<Vec<_>>()
        .join(",");
    if has_loop {
        PrefixAction::Escalate {
            reason: format!("prefix-risk loop: {reason} score={risk}"),
            councils: vec![COUNCIL_13],
        }
    } else {
        PrefixAction::Halt {
            reason: format!("prefix-risk halt: {reason} score={risk}"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrefixRunResult {
    pub snapshot: PrefixSnapshot,
    pub applied_count: usize,
    pub halted_at: Option<usize>,
    pub evidence_hashes: Vec<String>,
    pub harness_proposal: Option<HarnessDeltaProposal>,
}

impl PrefixRunResult {
    pub fn terminal_action(&self) -> &PrefixAction {
        &self.snapshot.action
    }

    pub fn did_escalate(&self) -> bool {
        matches!(self.snapshot.action, PrefixAction::Escalate { .. })
    }

    pub fn did_halt(&self) -> bool {
        matches!(
            self.snapshot.action,
            PrefixAction::Halt { .. }
        ) || self.snapshot.state == PrefixState::Halted
            || self.snapshot.state == PrefixState::Reverted
    }
}

fn emit_prefix_row(
    chain: &mut EvidenceChain,
    step: &TrajectoryStep,
    record: &PrefixStepRecord,
    actor: &str,
) -> Result<String, EvidenceError> {
    let directive = record.action.as_kind();
    let rec = chain.append(EvidenceDraft {
        subject: format!("prefix:{actor}:{}", record.index),
        input: record.payload_digest.clone(),
        claim: format!(
            "prefix {} risk={} state={}",
            step.kind.as_str(),
            record.risk_after,
            record.state.as_str()
        ),
        evidence_pointers: vec![
            format!("prefix-step:{}", record.index),
            format!("kind:{}", step.kind.as_str()),
            format!("findings:{}", record
                .findings
                .iter()
                .map(|f| f.as_label())
                .collect::<Vec<_>>()
                .join(",")),
        ],
        directive: directive.into(),
        scope: "lattice-conductor-v14".into(),
        kind: EvidenceKind::Prefix,
        decision: directive.into(),
        timestamp: record.index as u64,
        actor: actor.into(),
        auditor: "prefix-risk-face".into(),
    })?;
    Ok(rec.hash)
}

fn fire_live_step(
    api: &mut MercyGatedApi,
    arbitration: &CouncilArbitrationEngine,
    step: &TrajectoryStep,
) -> Option<(bool, Option<String>)> {
    match step.kind {
        TrajectoryStepKind::Search | TrajectoryStepKind::Read => {
            let resp = api.handle_request(
                MercyApiRequest {
                    kind: ApiRequestKind::HealthCheck,
                    payload: step.payload.clone(),
                    claimed_mercy: step.claimed_mercy,
                    actor: step.actor.clone(),
                },
                Some(arbitration),
            );
            Some((resp.accepted, resp.evidence_hash))
        }
        TrajectoryStepKind::Vote => {
            let resp = api.handle_request(
                MercyApiRequest {
                    kind: ApiRequestKind::CouncilQuery,
                    payload: step.payload.clone(),
                    claimed_mercy: step.claimed_mercy,
                    actor: step.actor.clone(),
                },
                Some(arbitration),
            );
            Some((resp.accepted, resp.evidence_hash))
        }
        TrajectoryStepKind::Wrap => {
            let resp = wrap_model_output(
                api,
                arbitration,
                ModelSurface::Other("prefix-trajectory"),
                &step.payload,
                step.claimed_mercy,
                &step.actor,
            );
            Some((resp.accepted, resp.evidence_hash))
        }
        TrajectoryStepKind::Edit | TrajectoryStepKind::Tool => {
            let label = if step.kind == TrajectoryStepKind::Edit {
                "prefix-edit"
            } else {
                "prefix-tool"
            };
            let resp = api.handle_request(
                MercyApiRequest {
                    kind: ApiRequestKind::Custom(label.into()),
                    payload: step.payload.clone(),
                    claimed_mercy: step.claimed_mercy,
                    actor: step.actor.clone(),
                },
                Some(arbitration),
            );
            Some((resp.accepted, resp.evidence_hash))
        }
        TrajectoryStepKind::Verify => {
            let ok = api
                .evidence_chain()
                .lock()
                .ok()
                .and_then(|c| c.verify().ok())
                .is_some();
            Some((ok, None))
        }
        TrajectoryStepKind::Revert => None,
    }
}

/// Lived multi-step path: score each prefix, fire wrap / handle_request only
/// while Continue, halt or escalate before later steps apply.
pub fn run_prefix_trajectory(
    api: &mut MercyGatedApi,
    arbitration: &CouncilArbitrationEngine,
    steps: &[TrajectoryStep],
    config: &PrefixConfig,
) -> PrefixRunResult {
    let mut applied_count = 0usize;
    let mut halted_at = None;
    let mut evidence_hashes = Vec::new();
    let mut filled_records: Vec<PrefixStepRecord> = Vec::new();
    let mut terminal = PrefixSnapshot {
        risk: 0,
        findings: vec![],
        state: PrefixState::Idle,
        action: PrefixAction::Continue,
        records: vec![],
    };

    for i in 0..steps.len() {
        let snap = score_prefix(&steps[..=i], config);
        let mut rec = snap
            .records
            .last()
            .cloned()
            .expect("score_prefix emits a record per step");
        let rule_blocks = rec.findings.iter().any(|f| f.blocks_live_apply());
        let mut layer0_accepted = None;

        if rec.action.is_stop() && rule_blocks {
            rec.layer0_accepted = None;
            if let Ok(mut chain) = api.evidence_chain().lock() {
                if let Ok(h) = emit_prefix_row(&mut chain, &steps[i], &rec, &steps[i].actor) {
                    rec.evidence_hash = Some(h.clone());
                    evidence_hashes.push(h);
                }
            }
            if matches!(rec.action, PrefixAction::Escalate { .. }) {
                escalate_to_council(api, &steps[i], &rec);
            }
            filled_records.push(rec);
            halted_at = Some(i);
            terminal = snap;
            terminal.records = filled_records.clone();
            break;
        }

        match fire_live_step(api, arbitration, &steps[i]) {
            Some((accepted, live_hash)) => {
                layer0_accepted = Some(accepted);
                if let Some(h) = live_hash {
                    evidence_hashes.push(h);
                }
                if accepted {
                    applied_count += 1;
                } else {
                    rec.action = PrefixAction::Halt {
                        reason: match &rec.action {
                            PrefixAction::Halt { reason } => reason.clone(),
                            _ => "Layer 0 / live path rejected prefix step".into(),
                        },
                    };
                    rec.state = PrefixState::Halted;
                }
            }
            None => {
                if steps[i].kind == TrajectoryStepKind::Revert {
                    rec.state = PrefixState::Reverted;
                    rec.action = PrefixAction::Halt {
                        reason: "revert — human override, no harness mutation".into(),
                    };
                }
            }
        }
        rec.layer0_accepted = layer0_accepted;

        if let Ok(mut chain) = api.evidence_chain().lock() {
            if let Ok(h) = emit_prefix_row(&mut chain, &steps[i], &rec, &steps[i].actor) {
                rec.evidence_hash = Some(h.clone());
                evidence_hashes.push(h);
            }
        }

        let stop = rec.action.is_stop();
        filled_records.push(rec.clone());
        terminal = snap;
        terminal.records = filled_records.clone();
        terminal.action = rec.action.clone();
        terminal.state = rec.state;
        terminal.risk = rec.risk_after;
        terminal.findings = rec.findings.clone();

        if stop {
            halted_at = Some(i);
            break;
        }
    }

    if matches!(terminal.action, PrefixAction::Continue)
        && terminal.state == PrefixState::Verifying
    {
        terminal.state = PrefixState::Complete;
    }

    PrefixRunResult {
        snapshot: terminal,
        applied_count,
        halted_at,
        evidence_hashes,
        harness_proposal: None,
    }
}

fn escalate_to_council(api: &MercyGatedApi, step: &TrajectoryStep, rec: &PrefixStepRecord) {
    let chain = api.evidence_chain();
    let Ok(mut chain) = chain.lock() else {
        return;
    };
    let request = PatsagiReviewRequest {
        topic: "prefix-risk escalate".into(),
        summary: format!(
            "prefix halt-escalate at step {} kind={} risk={}",
            rec.index,
            step.kind.as_str(),
            rec.risk_after
        ),
        mercy_impact_score: 0.99,
        requested_by: step.actor.clone(),
    };
    let _ = PatsagiCouncilSimulator::review_with_evidence(
        &request,
        &mut chain,
        "prefix-risk-face",
        rec.index as u64,
    );
}

impl MercyGatedApi {
    pub fn run_prefix_trajectory(
        &mut self,
        arbitration: &CouncilArbitrationEngine,
        steps: &[TrajectoryStep],
        config: &PrefixConfig,
    ) -> PrefixRunResult {
        run_prefix_trajectory(self, arbitration, steps, config)
    }
}

// ── Harness evolution: proposal object only. Never silent apply. ──────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HarnessTarget {
    SystemPrompt,
    RuleBank,
    ToolPolicy,
}

impl HarnessTarget {
    pub fn as_str(self) -> &'static str {
        match self {
            HarnessTarget::SystemPrompt => "system_prompt",
            HarnessTarget::RuleBank => "rule_bank",
            HarnessTarget::ToolPolicy => "tool_policy",
        }
    }
}

/// SHE-like proposed harness delta. Evidence-backed proposal only.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HarnessDeltaProposal {
    pub target: HarnessTarget,
    pub proposed_delta: String,
    pub evidence_pointers: Vec<String>,
    pub evidence_hash: Option<String>,
    /// Always false through the public API.
    pub applied: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum HarnessError {
    #[error("silent harness mutation is forbidden")]
    SilentMutationForbidden,
    #[error("missing evidence record — harness delta is not apply")]
    MissingEvidence,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GatedSubmitReceipt {
    pub accepted: bool,
    pub evidence_hash: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HarnessFileEditResult {
    Miss { path: String, reason: String },
    ProposedOnly { proposal: HarnessDeltaProposal },
}

/// Public propose path. `applied` is always false.
pub fn propose_harness_delta(
    target: HarnessTarget,
    proposed_delta: impl Into<String>,
    evidence_hash: Option<&str>,
) -> HarnessDeltaProposal {
    HarnessDeltaProposal {
        target,
        proposed_delta: proposed_delta.into(),
        evidence_pointers: vec![
            format!("harness:{}", target.as_str()),
            "prefix-risk".into(),
        ],
        evidence_hash: evidence_hash
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string()),
        applied: false,
    }
}

/// There is no silent apply. This function exists so the public API can
/// name the miss instead of mutating prompt / rule bank / tool policy.
pub fn apply_harness_delta(_proposal: &HarnessDeltaProposal) -> Result<(), HarnessError> {
    Err(HarnessError::SilentMutationForbidden)
}

/// Harness-file edit that skipped `submit_self_evolution_proposal_securely`
/// is a named miss (not apply). A gated receipt still only yields a proposal.
pub fn apply_harness_file_edit(
    path: &str,
    bytes: &str,
    gated: Option<&GatedSubmitReceipt>,
) -> HarnessFileEditResult {
    let gated_ok = gated
        .map(|g| g.accepted && g.evidence_hash.as_deref().map(str::trim).is_some_and(|s| !s.is_empty()))
        .unwrap_or(false);
    if !gated_ok {
        return HarnessFileEditResult::Miss {
            path: path.into(),
            reason: "not apply — harness-file edit did not cross submit_self_evolution_proposal_securely / apply-class"
                .into(),
        };
    }
    let hash = gated.and_then(|g| g.evidence_hash.clone());
    let proposal = propose_harness_delta(HarnessTarget::SystemPrompt, bytes, hash.as_deref());
    debug_assert!(!proposal.applied);
    HarnessFileEditResult::ProposedOnly { proposal }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn search(p: &str) -> TrajectoryStep {
        TrajectoryStep::new(TrajectoryStepKind::Search, p, 0.99)
    }
    fn read(p: &str) -> TrajectoryStep {
        TrajectoryStep::new(TrajectoryStepKind::Read, p, 0.99)
    }
    fn wrap(p: &str) -> TrajectoryStep {
        TrajectoryStep::new(TrajectoryStepKind::Wrap, p, 0.99)
    }
    fn verify(p: &str) -> TrajectoryStep {
        TrajectoryStep::new(TrajectoryStepKind::Verify, p, 0.99)
    }
    fn tool(id: &str, p: &str) -> TrajectoryStep {
        TrajectoryStep::new(TrajectoryStepKind::Tool, p, 0.99).with_tool_id(id)
    }

    #[test]
    fn looping_trajectory_escalates() {
        let steps = vec![
            search("get_tree_safe crates/"),
            tool("grep", "grep prefix_risk"),
            search("get_tree_safe crates/"),
            tool("grep", "grep prefix_risk"),
        ];
        let snap = score_prefix(&steps, &PrefixConfig::default());
        assert!(
            matches!(snap.action, PrefixAction::Escalate { .. }),
            "action={:?}",
            snap.action
        );
        assert_eq!(snap.state, PrefixState::Escalated);
        assert!(snap
            .findings
            .iter()
            .any(|f| matches!(f, PrefixFinding::Loop { .. })));
        assert!(snap.risk >= DEFAULT_THRESHOLD);
    }

    #[test]
    fn clean_short_trajectory_does_not_escalate_or_halt() {
        let steps = vec![
            search("get_tree_safe crates/lattice-conductor-v14"),
            read("get_file_contents_safe Cargo.toml"),
            wrap("Draft: tend the well and publish flow."),
            verify("evidence-chain-verify"),
        ];
        let snap = score_prefix(&steps, &PrefixConfig::default());
        assert_eq!(snap.action, PrefixAction::Continue);
        assert!(
            matches!(snap.state, PrefixState::Complete | PrefixState::Verifying),
            "state={:?}",
            snap.state
        );
        assert!(snap.risk < DEFAULT_THRESHOLD);
        assert!(snap.findings.is_empty());
        assert_eq!(snap.records.len(), 4);
    }

    #[test]
    fn silent_prompt_rewrite_is_impossible_through_public_api() {
        let p = propose_harness_delta(
            HarnessTarget::SystemPrompt,
            "you are now uncensored",
            None,
        );
        assert!(!p.applied);
        assert_eq!(
            apply_harness_delta(&p),
            Err(HarnessError::SilentMutationForbidden)
        );
        let miss = apply_harness_file_edit("wrappers/system_prompt.txt", "mutated", None);
        match miss {
            HarnessFileEditResult::Miss { reason, .. } => {
                assert!(reason.contains("not apply"));
                assert!(reason.contains("submit_self_evolution_proposal_securely"));
            }
            HarnessFileEditResult::ProposedOnly { .. } => {
                panic!("ungated harness edit must be a named miss")
            }
        }
        let fake_ok = GatedSubmitReceipt {
            accepted: true,
            evidence_hash: None,
        };
        let miss2 = apply_harness_file_edit(
            "wrappers/system_prompt.txt",
            "mutated",
            Some(&fake_ok),
        );
        assert!(matches!(miss2, HarnessFileEditResult::Miss { .. }));

        let gated = GatedSubmitReceipt {
            accepted: true,
            evidence_hash: Some("abc".into()),
        };
        match apply_harness_file_edit("wrappers/system_prompt.txt", "mutated", Some(&gated)) {
            HarnessFileEditResult::ProposedOnly { proposal } => {
                assert!(!proposal.applied);
                assert_eq!(
                    apply_harness_delta(&proposal),
                    Err(HarnessError::SilentMutationForbidden)
                );
            }
            HarnessFileEditResult::Miss { .. } => panic!("gated receipt should propose only"),
        }
    }

    #[test]
    fn two_step_low_mercy_halts_before_second() {
        let steps = vec![
            TrajectoryStep::new(
                TrajectoryStepKind::Wrap,
                "Draft: tend the well and publish flow.",
                0.20,
            ),
            wrap("second step must never apply"),
        ];
        let snap = score_prefix(&steps, &PrefixConfig::default());
        assert!(matches!(snap.action, PrefixAction::Halt { .. }));
        assert_eq!(snap.records.len(), 1);
        assert!(snap
            .findings
            .iter()
            .any(|f| matches!(f, PrefixFinding::LowClaimedMercy { .. })));
    }

    #[test]
    fn skipped_verify_halts() {
        let steps = vec![
            search("plan"),
            wrap("Draft: tend the well and publish flow."),
            wrap("second wrap without verify"),
        ];
        let snap = score_prefix(&steps, &PrefixConfig::default());
        assert!(matches!(snap.action, PrefixAction::Halt { .. }));
        assert!(snap
            .findings
            .iter()
            .any(|f| matches!(f, PrefixFinding::SkippedVerify { .. })));
    }

    #[test]
    fn tool_before_plan_halts() {
        let steps = vec![tool("grep", "grep before any search")];
        let snap = score_prefix(&steps, &PrefixConfig::default());
        assert!(matches!(snap.action, PrefixAction::Halt { .. }));
        assert!(snap
            .findings
            .iter()
            .any(|f| matches!(f, PrefixFinding::ToolBeforePlan { .. })));
    }

    #[test]
    fn repeated_identical_tool_halts() {
        let steps = vec![
            search("plan"),
            tool("grep", "same"),
            verify("ok"),
            tool("grep", "same"),
            tool("grep", "same"),
        ];
        let snap = score_prefix(&steps, &PrefixConfig::default());
        assert!(matches!(snap.action, PrefixAction::Halt { .. }));
        assert!(snap
            .findings
            .iter()
            .any(|f| matches!(f, PrefixFinding::RepeatedIdenticalTool { .. })));
    }

    #[test]
    fn observer_model_defaults_off_and_stub_is_not_an_llm() {
        let cfg = PrefixConfig::default();
        assert!(!cfg.observer_model);
        assert_eq!(cfg.observer_bonus(&[]), 0);
        let mut on = PrefixConfig::default();
        on.observer_model = true;
        assert_eq!(on.observer_bonus(&[search("x")]), 0);
        let steps = vec![
            search("get_tree_safe crates/lattice-conductor-v14"),
            read("get_file_contents_safe Cargo.toml"),
            wrap("Draft: tend the well and publish flow."),
            verify("evidence-chain-verify"),
        ];
        let snap = score_prefix(&steps, &on);
        assert_eq!(snap.action, PrefixAction::Continue);
        assert!(!snap
            .findings
            .iter()
            .any(|f| matches!(f, PrefixFinding::Observer { .. })));
    }

    #[test]
    fn fixture_is_offline_checkable_and_hash_stable() {
        let raw = include_str!("../fixtures/prefix_trajectory_v0.json");
        let fx: serde_json::Value = serde_json::from_str(raw).expect("fixture json");
        assert_eq!(fx["canonical_version"], CANONICAL_VERSION);

        let clean: Vec<TrajectoryStep> =
            serde_json::from_value(fx["clean_short"]["steps"].clone()).unwrap();
        let looping: Vec<TrajectoryStep> =
            serde_json::from_value(fx["looping"]["steps"].clone()).unwrap();
        let two: Vec<TrajectoryStep> =
            serde_json::from_value(fx["two_step_low_mercy"]["steps"].clone()).unwrap();

        let cfg = PrefixConfig::default();
        let clean_snap = score_prefix(&clean, &cfg);
        let loop_snap = score_prefix(&looping, &cfg);
        let two_snap = score_prefix(&two, &cfg);

        assert_eq!(clean_snap.action.as_kind(), "continue");
        assert_eq!(loop_snap.action.as_kind(), "escalate");
        assert_eq!(two_snap.action.as_kind(), "halt");
        assert_eq!(two_snap.records.len(), 1);

        let clean_hash = clean_snap.compute_hash();
        let again = score_prefix(&clean, &cfg).compute_hash();
        assert_eq!(clean_hash, again);
        assert_eq!(clean_hash, fx["clean_short"]["hash"]);
        assert_eq!(loop_snap.compute_hash(), fx["looping"]["hash"]);
        assert_eq!(two_snap.compute_hash(), fx["two_step_low_mercy"]["hash"]);
        assert_eq!(clean_hash.len(), 64);
    }
}
