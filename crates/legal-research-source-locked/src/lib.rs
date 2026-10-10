//! On-disk research sketch only.
//!
//! Source-locked helpers for public legal texts.
//! Not a legal product. Not a resolver. Not a workspace member.
//! Outputs are drafts requiring human review.
//! See PUBLIC_CLAIM.lock.md and the 2026-10-10 PATSAGi minute.

#![deny(missing_docs)]

/// Mandatory seal that every public draft must carry.
pub const DRAFT_SEAL: &str = "DRAFT — human review required. Not legal advice. Not a product. See PUBLIC_CLAIM.lock.md.";

/// Research-only result wrapper. Never a resolution or opinion.
#[derive(Debug, Clone)]
pub struct ResearchDraft {
    /// The source-locked analysis text.
    pub body: String,
    /// Always true for this crate.
    pub is_draft: bool,
    /// The mandatory seal.
    pub seal: &'static str,
}

impl ResearchDraft {
    /// Construct a sealed research draft. Does not claim resolution.
    pub fn new(body: impl Into<String>) -> Self {
        Self {
            body: body.into(),
            is_draft: true,
            seal: DRAFT_SEAL,
        }
    }
}

/// Placeholder for future source-locked citation extraction.
/// Returns the input unchanged until a later named motion implements it.
pub fn extract_citations_research(source: &str) -> ResearchDraft {
    ResearchDraft::new(format!(
        "Source-locked citation extraction not yet implemented.\n\nInput length: {} chars.\n\n{}",
        source.len(),
        DRAFT_SEAL
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn draft_is_always_sealed() {
        let d = ResearchDraft::new("test");
        assert!(d.is_draft);
        assert!(d.seal.contains("DRAFT"));
        assert!(d.seal.contains("Not legal advice"));
    }
}
