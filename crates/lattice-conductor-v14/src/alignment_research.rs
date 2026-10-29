//! AlignmentResearchCouncil — PATSAGi sandbox for gated research proposals.
//!
//! Parallel researcher stubs may propose alignment tests, SAE probes, or gate
//! refinements. They cannot merge, cannot write Layer 0 thresholds, and cannot
//! score their own homework. Every artifact attaches an Opportunity 3 evidence
//! row. External audit lives on `AlignmentAuditHarness` — the researcher
//! process has no write into that ledger.
//!
//! Promote / merge / threshold-edit always Reject without Council 13 or the
//! existing human-override path. Merge and Layer 0 writes Reject even then.
//! This is not a running alignment researcher. Not Combined AGSi. Not METR.
//!
//! Lived surface: `lattice-conductor-v14` under conductor arbitration.
//! Forest name: `crates/patsagi-councils/docs/ALIGNMENT_RESEARCH_COUNCIL.md`.
//! Contact: info@Rathor.ai. Independent of xAI.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::evidence_chain::{
    payload_digest, EvidenceChain, EvidenceDraft, EvidenceError, EvidenceKind,
};
use crate::ra_thor_mercy_gated_api::MercyGatedApi;
use crate::CouncilArbitrationEngine;

pub const CANONICAL_VERSION: &str = "RT-ALIGN-RES-v1";
pub const RESEARCH_COUNCIL_ID: &str = "alignment-research-council";
pub const AUDIT_ACTOR: &str = "alignment-audit-harness";
pub const EVIDENCE_FACE: &str = "alignment-evidence-face";
pub const COUNCIL_13: &str = "council-13";
pub const HUMAN_OVERRIDE: &str = "human-override";
/// Valence floor invariant. The sandbox cannot lower or rewrite it.
pub const LAYER_0_VALENCE_FLOOR: f64 = 0.999999;
pub const MAX_STEPS: u32 = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HypothesisKind {
    AlignmentTest,
    SaeProbe,
    GateRefinement,
}

impl HypothesisKind {
    pub fn as_str(self) -> &'static str {
        match self {
            HypothesisKind::AlignmentTest => "alignment_test",
            HypothesisKind::SaeProbe => "sae_probe",
            HypothesisKind::GateRefinement => "gate_refinement",
        }
    }
}

/// First audit dimension is deception. Over-refusal and APTD purity are
/// parallel stubs. Wrap-bypass is a Ra-Thor-native extra.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuditDimension {
    Deception,
    OverRefusal,
    AptdPurity,
    WrapBypass,
}

impl AuditDimension {
    pub fn as_str(self) -> &'static str {
        match self {
            AuditDimension::Deception => "deception",
            AuditDimension::OverRefusal => "over_refusal",
            AuditDimension::AptdPurity => "aptd_purity",
            AuditDimension::WrapBypass => "wrap_bypass",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProposalStatus {
    Proposed,
    Rejected,
    /// Council 13 / human-override escalation. Not accepted. Not merged.
    Escalated,
}

impl ProposalStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            ProposalStatus::Proposed => "proposed",
            ProposalStatus::Rejected => "rejected",
            ProposalStatus::Escalated => "escalated",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuditVerdict {
    Pass,
    Fail,
    Refine,
}

impl AuditVerdict {
    pub fn as_str(self) -> &'static str {
        match self {
            AuditVerdict::Pass => "pass",
            AuditVerdict::Fail => "fail",
            AuditVerdict::Refine => "refine",
        }
    }
}

/// Sandbox face. No Accepted variant — researchers cannot mark accepted.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SandboxDecision {
    Proposed,
    Escalated { reason: String },
    Rejected { reason: String },
}

impl SandboxDecision {
    pub fn is_rejected(&self) -> bool {
        matches!(self, SandboxDecision::Rejected { .. })
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum AlignmentError {
    #[error("unknown research ticket {0}")]
    UnknownTicket(String),
    #[error("step budget exceeded for ticket {ticket} (max {max})")]
    StepBudgetExceeded { ticket: String, max: u32 },
    #[error("researcher cannot write the external audit score")]
    ResearcherCannotWriteAudit,
    #[error("same actor on proposal and audit: {actor}")]
    SelfAudit { actor: String },
    #[error("missing evidence record — no promotion")]
    MissingEvidence,
    #[error(transparent)]
    Evidence(#[from] EvidenceError),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResearchTicket {
    pub id: String,
    pub researcher_actor: String,
    pub hypothesis: String,
    pub kind: HypothesisKind,
    pub dimension: AuditDimension,
    pub max_steps: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResearchProposal {
    pub ticket_id: String,
    pub researcher_actor: String,
    pub hypothesis: String,
    pub kind: HypothesisKind,
    pub dimension: AuditDimension,
    pub artifact: String,
    pub steps_used: u32,
    pub max_steps: u32,
    pub status: ProposalStatus,
    pub evidence_hash: String,
    /// Wrap receipt pointer when a stub wrap ran. Not a live model call.
    #[serde(default)]
    pub wrap_hash: Option<String>,
}

impl ResearchProposal {
    pub fn canonical_preimage(&self) -> String {
        let mut map = BTreeMap::new();
        map.insert("artifact", self.artifact.clone());
        map.insert("dimension", self.dimension.as_str().to_string());
        map.insert("evidence_hash", self.evidence_hash.clone());
        map.insert("hypothesis", self.hypothesis.clone());
        map.insert("kind", self.kind.as_str().to_string());
        map.insert("max_steps", self.max_steps.to_string());
        map.insert("researcher_actor", self.researcher_actor.clone());
        map.insert("status", self.status.as_str().to_string());
        map.insert("steps_used", self.steps_used.to_string());
        map.insert("ticket_id", self.ticket_id.clone());
        map.insert(
            "wrap_hash",
            self.wrap_hash.clone().unwrap_or_else(|| "none".into()),
        );
        let body = serde_json::to_string(&map).unwrap_or_default();
        format!("{CANONICAL_VERSION}\n{body}")
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuditScore {
    pub proposal_evidence_hash: String,
    pub dimension: AuditDimension,
    pub verdict: AuditVerdict,
    pub actor: String,
    pub notes: String,
}

/// Offline fixture: proposal body plus an audit the researcher cannot overwrite.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProposalAuditFixture {
    pub proposal: ResearchProposal,
    pub audit: AuditScore,
}

#[derive(Debug, Default, Clone)]
pub struct AuditLedger {
    scores: Vec<AuditScore>,
}

impl AuditLedger {
    pub fn scores(&self) -> &[AuditScore] {
        &self.scores
    }
}

/// External scorer. Distinct type from the researcher process.
#[derive(Debug, Default, Clone)]
pub struct AlignmentAuditHarness {
    ledger: AuditLedger,
}

impl AlignmentAuditHarness {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn ledger(&self) -> &AuditLedger {
        &self.ledger
    }

    /// Score a proposal. Writer and proposal actor must differ. Researcher
    /// handles cannot call this path successfully.
    pub fn score(
        &mut self,
        proposal: &ResearchProposal,
        writer: &str,
        verdict: AuditVerdict,
        notes: &str,
    ) -> Result<AuditScore, AlignmentError> {
        if is_researcher(writer) {
            return Err(AlignmentError::ResearcherCannotWriteAudit);
        }
        if writer == proposal.researcher_actor {
            return Err(AlignmentError::SelfAudit {
                actor: writer.to_string(),
            });
        }
        if writer != AUDIT_ACTOR && writer != COUNCIL_13 && writer != HUMAN_OVERRIDE {
            return Err(AlignmentError::SelfAudit {
                actor: writer.to_string(),
            });
        }
        if proposal.evidence_hash.trim().is_empty() {
            return Err(AlignmentError::MissingEvidence);
        }
        let score = AuditScore {
            proposal_evidence_hash: proposal.evidence_hash.clone(),
            dimension: proposal.dimension,
            verdict,
            actor: writer.to_string(),
            notes: notes.replace(['\n', '\r'], " "),
        };
        self.ledger.scores.push(score.clone());
        Ok(score)
    }
}

/// PATSAGi sandbox council. Tickets, not a swarm runtime.
#[derive(Debug, Clone)]
pub struct AlignmentResearchCouncil {
    tickets: Vec<ResearchTicket>,
    steps_used: BTreeMap<String, u32>,
    proposals: Vec<ResearchProposal>,
}

impl AlignmentResearchCouncil {
    pub fn with_parallel_stubs() -> Self {
        Self {
            tickets: parallel_stub_tickets(),
            steps_used: BTreeMap::new(),
            proposals: Vec::new(),
        }
    }

    pub fn tickets(&self) -> &[ResearchTicket] {
        &self.tickets
    }

    pub fn proposals(&self) -> &[ResearchProposal] {
        &self.proposals
    }

    pub fn ticket(&self, id: &str) -> Option<&ResearchTicket> {
        self.tickets.iter().find(|t| t.id == id)
    }

    /// Researcher emits a proposal. Attaches an Opportunity 3 evidence row.
    /// Does not write an audit score. Does not mark accepted.
    pub fn emit_proposal(
        &mut self,
        ticket_id: &str,
        artifact: &str,
        chain: &mut EvidenceChain,
        timestamp: u64,
        extra_pointers: &[String],
    ) -> Result<ResearchProposal, AlignmentError> {
        let ticket = self
            .ticket(ticket_id)
            .cloned()
            .ok_or_else(|| AlignmentError::UnknownTicket(ticket_id.into()))?;
        let used = self.steps_used.get(ticket_id).copied().unwrap_or(0);
        if used >= ticket.max_steps {
            return Err(AlignmentError::StepBudgetExceeded {
                ticket: ticket_id.into(),
                max: ticket.max_steps,
            });
        }
        let steps_used = used + 1;
        let mut pointers = vec![
            format!("ticket:{}", ticket.id),
            format!("dimension:{}", ticket.dimension.as_str()),
            "layer0-non-bypassable".into(),
            "opportunity-3-evidence".into(),
        ];
        pointers.extend(extra_pointers.iter().cloned());
        let rec = chain.append(EvidenceDraft {
            subject: format!("alignment_research:{}", ticket.id),
            input: payload_digest(artifact),
            claim: ticket.hypothesis.clone(),
            evidence_pointers: pointers,
            directive: "propose".into(),
            scope: RESEARCH_COUNCIL_ID.into(),
            kind: EvidenceKind::AlignmentResearch,
            decision: "propose".into(),
            timestamp,
            actor: ticket.researcher_actor.clone(),
            auditor: EVIDENCE_FACE.into(),
        })?;
        let proposal = ResearchProposal {
            ticket_id: ticket.id.clone(),
            researcher_actor: ticket.researcher_actor.clone(),
            hypothesis: ticket.hypothesis.clone(),
            kind: ticket.kind,
            dimension: ticket.dimension,
            artifact: artifact.replace(['\n', '\r'], " "),
            steps_used,
            max_steps: ticket.max_steps,
            status: ProposalStatus::Proposed,
            evidence_hash: rec.hash,
            wrap_hash: extra_pointers
                .iter()
                .find(|p| p.starts_with("wrap:"))
                .map(|p| p.trim_start_matches("wrap:").to_string()),
        };
        self.steps_used.insert(ticket_id.to_string(), steps_used);
        self.proposals.push(proposal.clone());
        Ok(proposal)
    }

    /// Stub wrap: fixture text through `wrap_model_output`, then emit.
    /// No live model. Compute stays a wrap + evidence append.
    pub fn emit_proposal_with_stub_wrap(
        &mut self,
        ticket_id: &str,
        artifact: &str,
        api: &mut MercyGatedApi,
        arbitration: &CouncilArbitrationEngine,
        timestamp: u64,
    ) -> Result<ResearchProposal, AlignmentError> {
        let wrap = crate::wrap_model_output::wrap_model_output(
            api,
            arbitration,
            crate::wrap_model_output::ModelSurface::Other("alignment-research-stub"),
            artifact,
            LAYER_0_VALENCE_FLOOR,
            self.ticket(ticket_id)
                .map(|t| t.researcher_actor.as_str())
                .unwrap_or(RESEARCH_COUNCIL_ID),
        );
        let mut pointers = Vec::new();
        if let Some(hash) = wrap.evidence_hash.as_deref() {
            pointers.push(format!("wrap:{hash}"));
        }
        if let Some(hash) = wrap.inspect_packet_hash.as_deref() {
            pointers.push(format!("inspect-packet:{hash}"));
        }
        let chain_handle = api.evidence_chain();
        let mut chain = chain_handle
            .lock()
            .map_err(|_| EvidenceError::LockPoisoned)?;
        self.emit_proposal(ticket_id, artifact, &mut chain, timestamp, &pointers)
    }

    /// Researcher API: always Reject. No Accepted status exists.
    pub fn mark_accepted(&self, actor: &str, _proposal: &ResearchProposal) -> SandboxDecision {
        SandboxDecision::Rejected {
            reason: format!(
                "{actor} cannot mark accepted — AlignmentResearchCouncil has no Accepted status"
            ),
        }
    }

    /// Promote is Reject without a live evidence row. Researcher cannot promote.
    /// Council 13 / human override may escalate — not accept, not merge.
    pub fn promote(
        &self,
        actor: &str,
        evidence_hash: Option<&str>,
        chain: &EvidenceChain,
    ) -> SandboxDecision {
        if chain.gate_side_effect(evidence_hash).is_err() {
            return SandboxDecision::Rejected {
                reason: "missing evidence — no promotion".into(),
            };
        }
        if is_researcher(actor) || actor == RESEARCH_COUNCIL_ID {
            return SandboxDecision::Rejected {
                reason: "researcher cannot promote or mark accepted".into(),
            };
        }
        if actor != COUNCIL_13 && actor != HUMAN_OVERRIDE {
            return SandboxDecision::Rejected {
                reason: "promote requires Council 13 or the existing override path".into(),
            };
        }
        SandboxDecision::Escalated {
            reason: "human review — not accepted, not merged, Layer 0 unchanged".into(),
        }
    }

    /// Always Reject. The sandbox does not write GitHub or touch `main`.
    pub fn merge_to_main(&self, actor: &str, from_sha: &str) -> SandboxDecision {
        if !is_commit_sha(from_sha) {
            return SandboxDecision::Rejected {
                reason: format!(
                    "{actor} merge / create_branch without a human-owned SHA path — Rejected"
                ),
            };
        }
        SandboxDecision::Rejected {
            reason: format!("{actor} cannot merge main — human / PATSAGi merge gate"),
        }
    }

    /// Always Reject. No GitHub write. Missing SHA is named in the reason.
    pub fn create_branch(&self, actor: &str, from_sha: &str) -> SandboxDecision {
        if !is_commit_sha(from_sha) {
            return SandboxDecision::Rejected {
                reason: format!("{actor} create_branch without a human-owned SHA path — Rejected"),
            };
        }
        SandboxDecision::Rejected {
            reason: format!(
                "{actor} cannot create_branch from this sandbox — GitHub writes stay off"
            ),
        }
    }

    /// Always Reject. Layer 0 thresholds are not writable here, including the
    /// valence floor. Council 13 cannot flip this Reject into a write.
    pub fn edit_tolc_threshold(&self, actor: &str, proposed: f64) -> SandboxDecision {
        let _ = proposed;
        SandboxDecision::Rejected {
            reason: format!(
                "{actor} cannot edit TOLC / Layer 0 thresholds (valence floor stays {LAYER_0_VALENCE_FLOOR})"
            ),
        }
    }

    /// Researcher handle cannot overwrite the external audit fixture.
    pub fn overwrite_audit_fixture(
        &self,
        fixture: &mut ProposalAuditFixture,
    ) -> Result<(), AlignmentError> {
        let _ = fixture;
        Err(AlignmentError::ResearcherCannotWriteAudit)
    }

    /// Researcher process must not write the audit ledger.
    pub fn write_audit(
        &self,
        _ledger: &mut AuditLedger,
        _score: AuditScore,
    ) -> Result<(), AlignmentError> {
        Err(AlignmentError::ResearcherCannotWriteAudit)
    }
}

fn parallel_stub_tickets() -> Vec<ResearchTicket> {
    vec![
        ResearchTicket {
            id: "hyp-deception-v0".into(),
            researcher_actor: "alignment-researcher-deception".into(),
            hypothesis: "A fluent wrap-looking string that never recorded a packet is not inspectable apply.".into(),
            kind: HypothesisKind::AlignmentTest,
            dimension: AuditDimension::Deception,
            max_steps: MAX_STEPS,
        },
        ResearchTicket {
            id: "hyp-over-refusal-v0".into(),
            researcher_actor: "alignment-researcher-over-refusal".into(),
            hypothesis: "Benign employ-loop draft text is still a draft; over-refusal is not a Layer 0 pass.".into(),
            kind: HypothesisKind::AlignmentTest,
            dimension: AuditDimension::OverRefusal,
            max_steps: MAX_STEPS,
        },
        ResearchTicket {
            id: "hyp-aptd-purity-v0".into(),
            researcher_actor: "alignment-researcher-aptd-purity".into(),
            hypothesis: "Council or researcher self-audit (actor == auditor) stays Rejected.".into(),
            kind: HypothesisKind::GateRefinement,
            dimension: AuditDimension::AptdPurity,
            max_steps: MAX_STEPS,
        },
        ResearchTicket {
            id: "hyp-wrap-bypass-v0".into(),
            researcher_actor: "alignment-researcher-wrap-bypass".into(),
            hypothesis: "A steering or SAE packet cannot flip Layer 0 Reject into Apply.".into(),
            kind: HypothesisKind::SaeProbe,
            dimension: AuditDimension::WrapBypass,
            max_steps: MAX_STEPS,
        },
    ]
}

pub fn is_researcher(actor: &str) -> bool {
    actor == RESEARCH_COUNCIL_ID || actor.starts_with("alignment-researcher")
}

fn is_commit_sha(s: &str) -> bool {
    let n = s.len();
    (n == 40 || n == 64) && s.chars().all(|c| c.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn emit_deception(council: &mut AlignmentResearchCouncil) -> (EvidenceChain, ResearchProposal) {
        let mut chain = EvidenceChain::new();
        let proposal = council
            .emit_proposal(
                "hyp-deception-v0",
                "Propose fixture: wrap without packet is not inspectable apply.",
                &mut chain,
                1,
                &[],
            )
            .expect("emit");
        (chain, proposal)
    }

    #[test]
    fn researcher_can_emit_proposal_with_evidence() {
        let mut council = AlignmentResearchCouncil::with_parallel_stubs();
        let ids: Vec<_> = council.tickets().iter().map(|t| t.id.as_str()).collect();
        assert!(ids.contains(&"hyp-deception-v0"));
        assert!(ids.contains(&"hyp-over-refusal-v0"));
        assert!(ids.contains(&"hyp-aptd-purity-v0"));
        let hypotheses: Vec<_> = council
            .tickets()
            .iter()
            .map(|t| t.hypothesis.as_str())
            .collect();
        assert_eq!(hypotheses.len(), 4);
        assert_eq!(
            hypotheses
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            4,
            "parallel stubs must start from distinct hypotheses"
        );

        let (chain, proposal) = emit_deception(&mut council);
        assert_eq!(proposal.status, ProposalStatus::Proposed);
        assert!(!proposal.evidence_hash.is_empty());
        chain
            .gate_side_effect(Some(&proposal.evidence_hash))
            .unwrap();
        assert_eq!(chain.records()[0].kind, EvidenceKind::AlignmentResearch);
        assert_ne!(chain.records()[0].actor, chain.records()[0].auditor);
        assert_eq!(proposal.dimension, AuditDimension::Deception);
        assert!(proposal.canonical_preimage().starts_with(CANONICAL_VERSION));
    }

    #[test]
    fn researcher_cannot_mark_accepted() {
        let mut council = AlignmentResearchCouncil::with_parallel_stubs();
        let (_chain, proposal) = emit_deception(&mut council);
        let decision = council.mark_accepted(&proposal.researcher_actor, &proposal);
        assert!(decision.is_rejected());
        assert_eq!(proposal.status, ProposalStatus::Proposed);
    }

    #[test]
    fn researcher_cannot_edit_tolc_thresholds() {
        let council = AlignmentResearchCouncil::with_parallel_stubs();
        let lower = council.edit_tolc_threshold("alignment-researcher-deception", 0.5);
        assert!(lower.is_rejected());
        let same = council.edit_tolc_threshold(COUNCIL_13, LAYER_0_VALENCE_FLOOR);
        assert!(same.is_rejected());
        let raise = council.edit_tolc_threshold(HUMAN_OVERRIDE, 1.0);
        assert!(raise.is_rejected());
    }

    #[test]
    fn missing_evidence_blocks_promotion() {
        let council = AlignmentResearchCouncil::with_parallel_stubs();
        let chain = EvidenceChain::new();
        let miss = council.promote(COUNCIL_13, None, &chain);
        assert!(miss.is_rejected());
        match miss {
            SandboxDecision::Rejected { reason } => {
                assert!(reason.contains("missing evidence"));
            }
            _ => panic!("expected reject"),
        }
    }

    #[test]
    fn researcher_cannot_promote_even_with_evidence() {
        let mut council = AlignmentResearchCouncil::with_parallel_stubs();
        let (chain, proposal) = emit_deception(&mut council);
        let decision = council.promote(
            &proposal.researcher_actor,
            Some(&proposal.evidence_hash),
            &chain,
        );
        assert!(decision.is_rejected());
        let escalated = council.promote(COUNCIL_13, Some(&proposal.evidence_hash), &chain);
        assert!(matches!(escalated, SandboxDecision::Escalated { .. }));
    }

    #[test]
    fn sandbox_merge_and_branch_without_sha_are_rejected() {
        let council = AlignmentResearchCouncil::with_parallel_stubs();
        let merge = council.merge_to_main("alignment-researcher-deception", "main");
        assert!(merge.is_rejected());
        let branch = council.create_branch(RESEARCH_COUNCIL_ID, "main");
        assert!(branch.is_rejected());
        match branch {
            SandboxDecision::Rejected { reason } => {
                assert!(reason.contains("SHA"));
            }
            _ => panic!("expected SHA reject"),
        }
        let sha = "a".repeat(40);
        assert!(council
            .create_branch("alignment-researcher-aptd-purity", &sha)
            .is_rejected());
        assert!(council.merge_to_main(COUNCIL_13, &sha).is_rejected());
    }

    #[test]
    fn same_actor_on_proposal_and_audit_is_rejected() {
        let mut council = AlignmentResearchCouncil::with_parallel_stubs();
        let (mut chain, proposal) = emit_deception(&mut council);
        let err = chain
            .append(EvidenceDraft {
                subject: "alignment_research:self-score".into(),
                input: payload_digest("self"),
                claim: "score own homework".into(),
                evidence_pointers: vec!["forbidden".into()],
                directive: "accept".into(),
                scope: RESEARCH_COUNCIL_ID.into(),
                kind: EvidenceKind::AlignmentResearch,
                decision: "accept".into(),
                timestamp: 2,
                actor: proposal.researcher_actor.clone(),
                auditor: proposal.researcher_actor.clone(),
            })
            .unwrap_err();
        assert!(matches!(err, EvidenceError::SelfAudit { .. }));

        let mut harness = AlignmentAuditHarness::new();
        let self_score = harness.score(
            &proposal,
            &proposal.researcher_actor,
            AuditVerdict::Pass,
            "recursive self-approval",
        );
        assert!(matches!(
            self_score,
            Err(AlignmentError::ResearcherCannotWriteAudit) | Err(AlignmentError::SelfAudit { .. })
        ));
    }

    #[test]
    fn external_harness_scores_and_researcher_cannot_overwrite_fixture() {
        let mut council = AlignmentResearchCouncil::with_parallel_stubs();
        let (_chain, proposal) = emit_deception(&mut council);
        let mut harness = AlignmentAuditHarness::new();
        let score = harness
            .score(
                &proposal,
                AUDIT_ACTOR,
                AuditVerdict::Refine,
                "External score. Researcher cannot write this field.",
            )
            .expect("external audit");
        assert_eq!(score.actor, AUDIT_ACTOR);
        assert_ne!(score.actor, proposal.researcher_actor);

        let mut fixture = ProposalAuditFixture {
            proposal: proposal.clone(),
            audit: score.clone(),
        };
        let original = fixture.audit.clone();
        assert_eq!(
            council.overwrite_audit_fixture(&mut fixture),
            Err(AlignmentError::ResearcherCannotWriteAudit)
        );
        assert_eq!(fixture.audit, original);
        assert_eq!(
            council.write_audit(&mut AuditLedger::default(), score.clone()),
            Err(AlignmentError::ResearcherCannotWriteAudit)
        );

        let raw = include_str!("../fixtures/alignment_research_proposal_audit_v0.json");
        let file_fixture: ProposalAuditFixture =
            serde_json::from_str(raw).expect("proposal+audit fixture");
        assert_eq!(file_fixture.audit.actor, AUDIT_ACTOR);
        assert_ne!(
            file_fixture.proposal.researcher_actor,
            file_fixture.audit.actor
        );
        let before = file_fixture.audit.clone();
        let mut file_fixture = file_fixture;
        assert!(council.overwrite_audit_fixture(&mut file_fixture).is_err());
        assert_eq!(file_fixture.audit, before);
    }

    #[test]
    fn step_budget_is_bounded() {
        let mut council = AlignmentResearchCouncil::with_parallel_stubs();
        let mut chain = EvidenceChain::new();
        for i in 0..MAX_STEPS {
            council
                .emit_proposal(
                    "hyp-over-refusal-v0",
                    &format!("step {i}"),
                    &mut chain,
                    i as u64 + 1,
                    &[],
                )
                .unwrap();
        }
        let err = council
            .emit_proposal("hyp-over-refusal-v0", "one more", &mut chain, 99, &[])
            .unwrap_err();
        assert!(matches!(err, AlignmentError::StepBudgetExceeded { .. }));
    }
}
