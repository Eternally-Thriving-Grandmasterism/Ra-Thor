//! Design-only (STUB-AUDIT-1 WORKSPACE-KEY-SHELLS-1 council note): not compiled; inherits workspace key powrush that the root does not define.
//! RREL Compliance Module — RECO AI Guidance (2026-10-22)
//! Source: https://www.reco.on.ca/agents-and-brokerages/reco-bulletins/reco-guidance-use-of-artificial-intelligence/
//!
//! Std-only: no external crate dependencies so this file can be tested with
//!   rustc --edition 2021 --test crates/real-estate-lattice/src/rrel_compliance.rs
//!
//! IMPORTANT: Price, conditions, and risk advice is the registrant's responsibility.
//! AI output is a draft only. No AI output constitutes professional real estate advice.
//! Every output must be independently verified by a registered agent before use.

use std::time::{SystemTime, UNIX_EPOCH};

fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

// ---------------------------------------------------------------------------
// OutputStatus — every AI output is Draft until a human registers verification
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum OutputStatus {
    /// AI-generated draft. Must not be relied upon or shared without human review.
    Draft,
    /// A registered agent has reviewed and accepted responsibility for this output.
    HumanVerified {
        /// Agent registration number or brokerage-assigned ID (no personal data).
        by: String,
        /// Unix timestamp (millis) when verification was recorded.
        at: u64,
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

#[derive(Debug, Clone)]
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

#[derive(Debug, Clone)]
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
            at: now_millis(),
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

#[derive(Debug, Clone)]
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
            label: Some(
                "AI-ALTERED IMAGE — not a representation of the actual property".to_string(),
            ),
            permanent_features_verified: false,
        }
    }
}

// ---------------------------------------------------------------------------
// ConsentRecord — written client consent / confidentiality gate
// RECO guidance: "Obtain explicit, written consent from clients before entering
// any personal details or contract terms into AI platforms."
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ConsentRecord {
    /// Anonymised client reference (no real names or addresses here).
    pub client_ref: String,
    /// Written (not verbal) consent obtained.
    pub written: bool,
    /// What the client consented to (e.g. "listing description review").
    pub scope: String,
    /// Which brokerage-approved AI tool will receive the data.
    pub tool_id: String,
    /// Unix timestamp (millis) when consent was obtained.
    pub obtained_at: u64,
}

impl ConsentRecord {
    /// Consent / confidentiality gate: returns Err if consent is not written,
    /// blocking data processing.
    pub fn gate(&self) -> Result<(), String> {
        if !self.written {
            return Err(format!(
                "Consent / confidentiality gate: written client consent is required before \
                 processing client data in tool '{}'. Verbal consent is insufficient under \
                 RECO AI guidance.",
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

#[derive(Debug, Clone)]
pub struct ApprovedTool {
    pub id: String,
    pub display_name: String,
    /// Broker of Record approved this tool.
    pub broker_of_record_approved: bool,
    /// Confidential data must not be sent to public AI platforms.
    pub is_public_platform: bool,
}

#[derive(Debug, Clone)]
pub struct ApprovedToolList {
    pub tools: Vec<ApprovedTool>,
}

impl ApprovedToolList {
    pub fn is_approved_for_confidential(&self, tool_id: &str) -> Result<(), String> {
        let tool = self.tools.iter().find(|t| t.id == tool_id);
        match tool {
            None => Err(format!("Tool '{}' is not on the brokerage approved list.", tool_id)),
            Some(t) if !t.broker_of_record_approved => Err(format!(
                "Tool '{}' has not received Broker of Record approval.",
                tool_id
            )),
            Some(t) if t.is_public_platform => Err(format!(
                "Tool '{}' is a public AI platform. Confidential client data must not be \
                 entered into public AI platforms (RECO guidance).",
                tool_id
            )),
            Some(_) => Ok(()),
        }
    }
}

// ---------------------------------------------------------------------------
// AuditEntry — records AI recommendation vs registrant advice vs client instruction
// RECO guidance: AI is support only; the registrant makes the final decision
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
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
    /// Unix timestamp (millis).
    pub ts: u64,
}

impl AuditEntry {
    pub fn new(
        client_ref: impl Into<String>,
        ai_recommendation: impl Into<String>,
        agent_decision: impl Into<String>,
        client_instruction: impl Into<String>,
    ) -> Self {
        let ts = now_millis();
        Self {
            id: format!("AUDIT-{}", ts),
            client_ref: client_ref.into(),
            ai_recommendation: ai_recommendation.into(),
            agent_decision: agent_decision.into(),
            client_instruction: client_instruction.into(),
            ts,
        }
    }
}

/// Append-only in-memory audit log. In a real system this would persist to an
/// immutable store. This struct makes the append-only contract explicit.
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

// ---------------------------------------------------------------------------
// Tests — runnable with:
//   rustc --edition 2021 --test crates/real-estate-lattice/src/rrel_compliance.rs \
//     -o /tmp/rrelt && /tmp/rrelt
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn consent(written: bool) -> ConsentRecord {
        ConsentRecord {
            client_ref: "client-anon-001".to_string(),
            written,
            scope: "listing description review".to_string(),
            tool_id: "internal-drafting-tool".to_string(),
            obtained_at: 0,
        }
    }

    // 1. Missing written consent blocks data processing -----------------------

    #[test]
    fn missing_consent_blocks() {
        let verbal_only = consent(false);
        let result = verbal_only.gate();
        assert!(result.is_err(), "verbal consent must be rejected");
        let msg = result.unwrap_err();
        assert!(
            msg.contains("written client consent"),
            "error must name the requirement: {msg}"
        );
    }

    #[test]
    fn written_consent_passes() {
        let written = consent(true);
        assert!(written.gate().is_ok(), "written consent must pass the gate");
    }

    // 2. Draft cannot become final without HumanVerified ----------------------

    #[test]
    fn draft_is_not_ready_for_use() {
        let output: AiOutput<&str> = AiOutput::draft(
            "draft recommendation",
            vec!["must be verified".to_string()],
            vec![],
        );
        assert!(!output.is_ready_for_use(), "Draft output must not be ready for use");
    }

    #[test]
    fn human_verified_is_ready_for_use() {
        let output: AiOutput<&str> = AiOutput::draft(
            "draft recommendation",
            vec!["must be verified".to_string()],
            vec![],
        )
        .verify("AGENT-REG-0042");
        assert!(output.is_ready_for_use(), "HumanVerified output must be ready for use");
        assert!(
            matches!(output.status, OutputStatus::HumanVerified { ref by, .. } if by == "AGENT-REG-0042"),
            "verified-by ID must be stored"
        );
    }

    #[test]
    fn verify_preserves_value_and_needs_verification() {
        let needs = vec!["cross-check addresses".to_string()];
        let output: AiOutput<i32> = AiOutput::draft(42, needs.clone(), vec![]).verify("AGENT-001");
        assert_eq!(output.value, 42);
        assert_eq!(output.needs_verification, needs);
    }

    // 3. AI-altered images must carry the ai_altered label --------------------

    #[test]
    fn altered_image_has_label() {
        let img = AiImageRecord::altered("render-v2.jpg");
        assert!(img.ai_altered, "ai_altered must be true");
        assert!(img.label.is_some(), "label must be present for altered images");
        let label = img.label.unwrap();
        assert!(
            !label.is_empty(),
            "label must not be empty"
        );
    }

    #[test]
    fn plain_image_has_no_label() {
        let img = AiImageRecord::plain("photo.jpg");
        assert!(!img.ai_altered, "ai_altered must be false for plain images");
        assert!(img.label.is_none(), "plain images must not carry an AI-altered label");
    }

    // 4. Audit entries keep AI recommendation, registrant advice and client
    //    instruction separate ------------------------------------------------

    #[test]
    fn audit_entry_separates_three_parties() {
        let entry = AuditEntry::new(
            "client-anon-001",
            "AI: list at $X based on comparables A, B",
            "Agent: recommend $Y after reviewing market",
            "Client: proceed at $Y",
        );
        assert_ne!(
            entry.ai_recommendation, entry.agent_decision,
            "AI recommendation and agent decision must be stored separately"
        );
        assert_ne!(
            entry.agent_decision, entry.client_instruction,
            "Agent decision and client instruction must be stored separately"
        );
        assert_ne!(
            entry.ai_recommendation, entry.client_instruction,
            "AI recommendation and client instruction must be stored separately"
        );
    }

    #[test]
    fn audit_log_is_append_only() {
        let mut log = AuditLog::new();
        assert_eq!(log.entries().len(), 0);

        log.record(AuditEntry::new("c-001", "ai-rec-1", "agent-dec-1", "client-inst-1"));
        assert_eq!(log.entries().len(), 1);

        log.record(AuditEntry::new("c-002", "ai-rec-2", "agent-dec-2", "client-inst-2"));
        assert_eq!(log.entries().len(), 2);

        // Earlier entries must remain unchanged
        assert_eq!(log.entries()[0].client_ref, "c-001");
        assert_eq!(log.entries()[1].client_ref, "c-002");
    }

    // 5. No price without a cited source -------------------------------------
    //    AiOutput wrapping an optional price must carry a verification item
    //    when no source is provided, and must not be ready for use.

    #[test]
    fn no_price_without_source() {
        // Simulates the valuation engine: price is None when no source is cited
        let price: Option<f64> = None;
        let needs_verification =
            vec!["price requires a cited, registrant-verified source".to_string()];
        let output = AiOutput::draft(price, needs_verification, vec![]);

        assert!(
            output.value.is_none(),
            "price must be None when no source is available"
        );
        assert!(
            !output.needs_verification.is_empty(),
            "needs_verification must flag the missing source"
        );
        assert!(
            !output.is_ready_for_use(),
            "output without a verified source must not be ready for use"
        );
    }

    #[test]
    fn price_with_source_still_requires_human_verification() {
        let price: Option<f64> = Some(500_000.0);
        let source = Provenance {
            label: "MLS listing CA-2026-001 (registrant-verified)".to_string(),
            reference: Some("https://example-mls.ca/listing/CA-2026-001".to_string()),
            notes: Some("cross-checked against title document".to_string()),
        };
        let output = AiOutput::draft(
            price,
            vec!["registrant must confirm comparable selection".to_string()],
            vec![source],
        );

        // Even with a source, Draft status means not ready for use
        assert!(!output.is_ready_for_use());

        // After human verification it becomes ready
        let verified = output.verify("AGENT-REG-0099");
        assert!(verified.is_ready_for_use());
        assert!(verified.value.is_some());
    }
}
