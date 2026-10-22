//! Design-only (STUB-AUDIT-1 WORKSPACE-KEY-SHELLS-1 council note): not compiled; inherits workspace key powrush that the root does not define.
//! RREL Compliance Module — RECO AI Guidance (2026-10-22)
//! Source: https://www.reco.on.ca/agents-and-brokerages/reco-bulletins/reco-guidance-use-of-artificial-intelligence/
//!
//! IMPORTANT: Price, conditions, and risk advice is the registrant's responsibility.
//! AI output is a draft only. No AI output constitutes professional real estate advice.
//! Every output must be independently verified by a registered agent before use.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// OutputStatus — every AI output is Draft until a human registers verification
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum OutputStatus {
    /// AI-generated draft. Must not be relied upon or shared without human review.
    Draft,
    /// A registered agent has reviewed and accepted responsibility for this output.
    HumanVerified {
        /// Agent registration number or brokerage-assigned ID (no personal data).
        by: String,
        at: DateTime<Utc>,
    },
}

impl OutputStatus {
    pub fn is_verified(&self) -> bool {
        matches!(self, OutputStatus::HumanVerified { .. })
    }
}

// ---------------------------------------------------------------------------
// Provenance — cited source required for every claim
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Provenance {
    /// Human-readable description of the source (e.g. "MLS listing ID CA-2026-001").
    pub label: String,
    /// Optional URL or document reference.
    pub reference: Option<String>,
    /// Verifier notes (e.g. "cross-checked against title document").
    pub notes: Option<String>,
}

// ---------------------------------------------------------------------------
// AiOutput<T> — wrapper that every AI recommendation/form/valuation must use
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiOutput<T> {
    pub value: T,
    pub status: OutputStatus,
    /// Items that must be independently verified before this output is used.
    pub needs_verification: Vec<String>,
    /// Cited sources for any facts contained in `value`.
    pub sources: Vec<Provenance>,
}

impl<T> AiOutput<T> {
    /// Create a new Draft output with mandatory verification items.
    pub fn draft(value: T, needs_verification: Vec<String>, sources: Vec<Provenance>) -> Self {
        Self {
            value,
            status: OutputStatus::Draft,
            needs_verification,
            sources,
        }
    }

    /// Upgrade to HumanVerified. Only a registered agent should call this.
    pub fn verify(mut self, agent_id: impl Into<String>) -> Self {
        self.status = OutputStatus::HumanVerified {
            by: agent_id.into(),
            at: Utc::now(),
        };
        self
    }

    pub fn is_ready_for_use(&self) -> bool {
        self.status.is_verified()
    }
}

// ---------------------------------------------------------------------------
// AI Image label — RECO guidance: label any digitally altered images
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiImageRecord {
    pub file_ref: String,
    /// True when any AI or digital editing was applied to this image.
    pub ai_altered: bool,
    /// Visible label that must accompany the image if ai_altered is true.
    pub label: Option<String>,
    /// Permanent physical features must not be fabricated.
    /// True when the image has been checked against verifiable property documents.
    pub permanent_features_verified: bool,
}

impl AiImageRecord {
    pub fn plain(file_ref: impl Into<String>) -> Self {
        Self {
            file_ref: file_ref.into(),
            ai_altered: false,
            label: None,
            permanent_features_verified: false,
        }
    }

    pub fn altered(file_ref: impl Into<String>) -> Self {
        Self {
            file_ref: file_ref.into(),
            ai_altered: true,
            label: Some("AI-ALTERED IMAGE — not a representation of the actual property".to_string()),
            permanent_features_verified: false,
        }
    }
}

// ---------------------------------------------------------------------------
// ConsentRecord — written client consent required before processing client data
// RECO guidance: "Obtain explicit, written consent from clients before entering
// any personal details or contract terms into AI platforms."
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsentRecord {
    /// Anonymised client reference (no real names or addresses here).
    pub client_ref: String,
    /// Written (not verbal) consent obtained.
    pub written: bool,
    /// What the client consented to (e.g. "listing description review").
    pub scope: String,
    /// Which brokerage-approved AI tool will receive the data.
    pub tool_id: String,
    pub obtained_at: DateTime<Utc>,
}

impl ConsentRecord {
    /// Returns Err if consent is not written, blocking data processing.
    pub fn gate(&self) -> Result<(), String> {
        if !self.written {
            return Err(format!(
                "Consent gate: written client consent is required before processing \
                 client data in tool '{}'. Verbal consent is insufficient under RECO \
                 AI guidance.",
                self.tool_id
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ApprovedToolList — brokerages must maintain a list of approved AI tools
// RECO guidance: "Brokerages should establish … an approved tool list"
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovedTool {
    pub id: String,
    pub display_name: String,
    /// Broker of Record approved this tool.
    pub broker_of_record_approved: bool,
    /// Confidential data must not be sent to public AI platforms.
    pub is_public_platform: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovedToolList {
    pub tools: Vec<ApprovedTool>,
}

impl ApprovedToolList {
    pub fn is_approved_for_confidential(&self, tool_id: &str) -> Result<(), String> {
        let tool = self.tools.iter().find(|t| t.id == tool_id);
        match tool {
            None => Err(format!(
                "Tool '{}' is not on the brokerage approved list.", tool_id
            )),
            Some(t) if !t.broker_of_record_approved => Err(format!(
                "Tool '{}' has not received Broker of Record approval.", tool_id
            )),
            Some(t) if t.is_public_platform => Err(format!(
                "Tool '{}' is a public AI platform. Confidential client data must \
                 not be entered into public AI platforms (RECO guidance).", tool_id
            )),
            Some(_) => Ok(()),
        }
    }
}

// ---------------------------------------------------------------------------
// AuditEntry — records AI recommendation vs registrant advice vs client instruction
// RECO guidance: AI is support only; the registrant makes the final decision
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    pub id: String,
    /// Anonymised client reference.
    pub client_ref: String,
    /// What the AI system suggested.
    pub ai_recommendation: String,
    /// What the registrant advised the client (may differ from AI suggestion).
    pub agent_decision: String,
    /// What the client ultimately instructed.
    pub client_instruction: String,
    pub ts: DateTime<Utc>,
}

impl AuditEntry {
    pub fn new(
        client_ref: impl Into<String>,
        ai_recommendation: impl Into<String>,
        agent_decision: impl Into<String>,
        client_instruction: impl Into<String>,
    ) -> Self {
        Self {
            id: format!("AUDIT-{}", Utc::now().timestamp_micros()),
            client_ref: client_ref.into(),
            ai_recommendation: ai_recommendation.into(),
            agent_decision: agent_decision.into(),
            client_instruction: client_instruction.into(),
            ts: Utc::now(),
        }
    }
}

/// Append-only in-memory audit log. In a real system this would persist to
/// an immutable store. This struct makes the append-only contract explicit.
#[derive(Debug, Default)]
pub struct AuditLog {
    entries: Vec<AuditEntry>,
}

impl AuditLog {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record(&mut self, entry: AuditEntry) {
        self.entries.push(entry);
    }

    pub fn entries(&self) -> &[AuditEntry] {
        &self.entries
    }
}
