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

mod citations;

#[doc(inline)]
pub use citations::find_reporter_citations;

/// Copy year / uppercase-reporter / number spans from `source` into a sealed draft.
///
/// Pattern: `\d{4}\s+[A-Z]+\s+\d+`, with boundaries so a match is not sliced out of a longer token.
/// The body lists verbatim spans from the supplied text. Volume-reporter-page forms stay outside this pattern.
///
/// DRAFT — human review required. Not legal advice. Not a product. See PUBLIC_CLAIM.lock.md.
pub fn extract_citations_research(source: &str) -> ResearchDraft {
    let hits = find_reporter_citations(source);
    let listed = if hits.is_empty() {
        "(no matches)".to_string()
    } else {
        hits.join("\n")
    };
    ResearchDraft::new(format!(
        "Source-locked reporter citation matches.\nPattern only. Spans are copied from the supplied text.\n\n{listed}\n\n{DRAFT_SEAL}"
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

    #[test]
    fn empty_source_draft_stays_sealed() {
        let d = extract_citations_research("");
        assert!(d.is_draft);
        assert_eq!(d.seal, DRAFT_SEAL);
        assert!(d.body.contains("(no matches)"));
        assert!(d.body.contains(DRAFT_SEAL));
    }
}
