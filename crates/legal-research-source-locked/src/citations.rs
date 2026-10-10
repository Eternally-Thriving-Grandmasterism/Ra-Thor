//! Source-locked scan for year / reporter / number spans.
//!
//! DRAFT — human review required. Not legal advice. Not a product. See PUBLIC_CLAIM.lock.md.

use regex::Regex;
use std::sync::OnceLock;

/// Pattern named by the 2026-10-10 task card, plus token boundaries.
///
/// Boundaries keep a match from being sliced out of a longer digit or letter run.
const REPORTER_CITATION_PATTERN: &str = r"\b\d{4}\s+[A-Z]+\s+\d+\b";

fn reporter_citation_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(REPORTER_CITATION_PATTERN)
            .expect("reporter citation pattern is a valid regex")
    })
}

/// Copy each year / uppercase-reporter / number span from `source`, in order.
///
/// The span is the regex match, including the whitespace that appeared in the source.
/// Repeated spans are kept. Lowercase reporter tokens stay in the source unread by this scan.
/// The result is a list of copied spans.
///
/// DRAFT — human review required. Not legal advice. Not a product. See PUBLIC_CLAIM.lock.md.
pub fn find_reporter_citations(source: &str) -> Vec<String> {
    reporter_citation_re()
        .find_iter(source)
        .map(|hit| hit.as_str().to_string())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lowercase_and_glued_digits_stay_out() {
        let hits = find_reporter_citations("see 2024 scc 15 and 12024 SCC 15 and 2024 SCC15.");
        assert!(hits.is_empty());
    }
}
