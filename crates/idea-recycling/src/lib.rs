//! Idea recycling: normalize base ideas, extract themes, and pair them.
//!
//! Pairings are text operations. They are seeds for a later check.
//! Process: recycle, then cross-pollinate. See
//! `docs/science/INNOVATIONS-GENERATOR-RUN-2026-09-07-TOLC-BASE-REALITY.md`.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

/// Hard cap on ideas kept after recycling, before pairing.
/// Ported from the archive extractor's cap of 40.
pub const MAX_RECYCLED: usize = 40;

/// One base idea from the caller.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BaseIdea {
    pub text: String,
    #[serde(default)]
    pub sources: Vec<String>,
    /// Caller-supplied note. Empty means no evidence note was attached.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evidence: Option<String>,
}

impl BaseIdea {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            sources: Vec::new(),
            evidence: None,
        }
    }

    pub fn with_sources(mut self, sources: Vec<String>) -> Self {
        self.sources = sources;
        self
    }

    pub fn with_evidence(mut self, evidence: impl Into<String>) -> Self {
        self.evidence = Some(evidence.into());
        self
    }
}

/// A normalized idea with themes taken from its own text.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecycledIdea {
    pub text: String,
    pub themes: Vec<String>,
    pub sources: Vec<String>,
    pub evidence: Option<String>,
}

/// One or two recycled ideas that can become a seed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pairing {
    pub ideas: Vec<RecycledIdea>,
    pub shared_themes: Vec<String>,
    pub score: u64,
}

/// Crate surface for recycle → cross-pollinate.
#[derive(Clone, Copy, Debug, Default)]
pub struct IdeaRecycler;

impl IdeaRecycler {
    pub fn new() -> Self {
        Self
    }

    pub fn recycle(&self, ideas: &[BaseIdea]) -> Vec<RecycledIdea> {
        recycle(ideas)
    }

    pub fn cross_pollinate(&self, ideas: &[RecycledIdea]) -> Vec<Pairing> {
        cross_pollinate(ideas)
    }
}

/// Drop empty text, merge duplicates, keep the strongest 40.
pub fn recycle(ideas: &[BaseIdea]) -> Vec<RecycledIdea> {
    let mut recycled: Vec<RecycledIdea> = Vec::new();
    for idea in ideas {
        let text = collapse_whitespace(idea.text.trim());
        if text.is_empty() {
            continue;
        }
        let key = normalize_key(&text);
        let sources = clean_sources(&idea.sources);
        let evidence = clean_evidence(idea.evidence.as_deref());
        if let Some(existing) = recycled
            .iter_mut()
            .find(|item| normalize_key(&item.text) == key)
        {
            merge_sources(&mut existing.sources, &sources);
            if existing.evidence.is_none() {
                existing.evidence = evidence;
            }
            continue;
        }
        recycled.push(RecycledIdea {
            themes: extract_themes(&text),
            text,
            sources,
            evidence,
        });
    }
    recycled.sort_by(|a, b| signal(b).cmp(&signal(a)).then_with(|| a.text.cmp(&b.text)));
    recycled.truncate(MAX_RECYCLED);
    recycled
}

/// Pair distinct ideas. A single idea is returned as a one-idea pairing.
///
/// Order is deterministic: higher score first, then idea text, then sources.
pub fn cross_pollinate(ideas: &[RecycledIdea]) -> Vec<Pairing> {
    let mut pairings = Vec::new();
    if ideas.len() <= 1 {
        if let Some(idea) = ideas.first() {
            pairings.push(single_pairing(idea));
        }
        return pairings;
    }
    for i in 0..ideas.len() {
        for j in (i + 1)..ideas.len() {
            pairings.push(pair_two(&ideas[i], &ideas[j]));
        }
    }
    pairings.sort_by(|a, b| {
        b.score
            .cmp(&a.score)
            .then_with(|| pairing_label(a).cmp(&pairing_label(b)))
    });
    pairings
}

fn single_pairing(idea: &RecycledIdea) -> Pairing {
    Pairing {
        score: signal(idea) as u64,
        shared_themes: idea.themes.clone(),
        ideas: vec![idea.clone()],
    }
}

fn pair_two(left: &RecycledIdea, right: &RecycledIdea) -> Pairing {
    let (first, second) = if left.text <= right.text {
        (left, right)
    } else {
        (right, left)
    };
    let shared_themes = intersect_themes(&first.themes, &second.themes);
    let sources = first.sources.len() + second.sources.len();
    let text_len = first.text.chars().count() + second.text.chars().count();
    let score = shared_themes.len() as u64 * 10 + sources as u64 * 2 + (text_len as u64) / 40;
    Pairing {
        ideas: vec![first.clone(), second.clone()],
        shared_themes,
        score,
    }
}

fn pairing_label(pairing: &Pairing) -> String {
    let texts = pairing
        .ideas
        .iter()
        .map(|idea| idea.text.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    let sources = pairing
        .ideas
        .iter()
        .flat_map(|idea| idea.sources.iter().map(String::as_str))
        .collect::<Vec<_>>()
        .join("\n");
    format!("{texts}\n{sources}")
}

fn signal(idea: &RecycledIdea) -> usize {
    idea.themes.len() * 3 + idea.text.chars().count() / 40
}

fn intersect_themes(left: &[String], right: &[String]) -> Vec<String> {
    let right_set: BTreeSet<&str> = right.iter().map(String::as_str).collect();
    left.iter()
        .filter(|theme| right_set.contains(theme.as_str()))
        .cloned()
        .collect()
}

fn collapse_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn normalize_key(text: &str) -> String {
    tokens(text)
        .into_iter()
        .take(12)
        .collect::<Vec<_>>()
        .join(" ")
}

fn clean_sources(sources: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    for source in sources {
        let trimmed = collapse_whitespace(source.trim());
        if trimmed.is_empty() || out.iter().any(|existing: &String| existing == &trimmed) {
            continue;
        }
        out.push(trimmed);
    }
    out.sort();
    out
}

fn merge_sources(into: &mut Vec<String>, extra: &[String]) {
    for source in extra {
        if !into.iter().any(|existing| existing == source) {
            into.push(source.clone());
        }
    }
    into.sort();
}

fn clean_evidence(evidence: Option<&str>) -> Option<String> {
    evidence
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .map(collapse_whitespace)
}

fn extract_themes(text: &str) -> Vec<String> {
    let mut themes = BTreeSet::new();
    for token in tokens(text) {
        if is_stopword(&token) || !token.chars().any(|ch| ch.is_alphabetic()) {
            continue;
        }
        if is_lexicon(&token) || token.len() >= 5 {
            themes.insert(token);
        }
    }
    themes.into_iter().take(12).collect()
}

fn tokens(text: &str) -> Vec<String> {
    text.split(|ch: char| !ch.is_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(|token| token.to_lowercase())
        .collect()
}

fn is_stopword(token: &str) -> bool {
    STOPWORDS.binary_search(&token).is_ok()
}

fn is_lexicon(token: &str) -> bool {
    LEXICON.binary_search(&token).is_ok()
}

/// Sorted. Short domain words that should still count as themes.
const LEXICON: &[&str] = &[
    "abundance",
    "claim",
    "conductor",
    "council",
    "crate",
    "displacement",
    "evidence",
    "feedback",
    "innovation",
    "lattice",
    "mercy",
    "proof",
    "reality",
    "recycling",
    "revenue",
    "source",
    "telemetry",
    "tolc",
];

/// Sorted. Common words that must not glue unrelated ideas together.
const STOPWORDS: &[&str] = &[
    "about", "above", "after", "again", "against", "being", "below", "between", "could", "doing",
    "during", "every", "first", "from", "have", "into", "other", "should", "their", "there",
    "these", "those", "under", "until", "where", "which", "while", "would", "your",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drops_blank_text_and_merges_duplicates() {
        let ideas = vec![
            BaseIdea::new("  Mercy checks   on the lattice  ").with_sources(vec!["a.md".into()]),
            BaseIdea::new("mercy checks on the lattice").with_sources(vec!["b.md".into()]),
            BaseIdea::new("   "),
        ];
        let recycled = recycle(&ideas);
        assert_eq!(recycled.len(), 1);
        assert_eq!(
            recycled[0].sources,
            vec!["a.md".to_string(), "b.md".to_string()]
        );
        assert!(recycled[0].themes.iter().any(|theme| theme == "mercy"));
        assert!(recycled[0].themes.iter().any(|theme| theme == "lattice"));
    }

    #[test]
    fn shared_theme_outranks_an_unrelated_pair() {
        let ideas = recycle(&[
            BaseIdea::new("telemetry seal for feedback events in the lattice"),
            BaseIdea::new("telemetry export for feedback reviews"),
            BaseIdea::new("paint colors for the office kitchen"),
        ]);
        let pairings = cross_pollinate(&ideas);
        assert!(pairings.len() >= 2);
        let top = &pairings[0];
        assert!(top.ideas.iter().all(|idea| idea.text.contains("telemetry")));
        assert!(top
            .shared_themes
            .iter()
            .any(|theme| theme == "telemetry" || theme == "feedback"));
        assert!(pairings[1..]
            .iter()
            .all(|pairing| pairing.score < top.score));
    }

    #[test]
    fn cross_pollinate_is_stable_when_input_order_changes() {
        let forward = recycle(&[
            BaseIdea::new("revenue path for listings this year"),
            BaseIdea::new("displacement concern in public service work"),
        ]);
        let backward = recycle(&[
            BaseIdea::new("displacement concern in public service work"),
            BaseIdea::new("revenue path for listings this year"),
        ]);
        assert_eq!(cross_pollinate(&forward), cross_pollinate(&backward));
        assert_eq!(cross_pollinate(&forward), cross_pollinate(&forward));
    }

    #[test]
    fn recycle_keeps_at_most_forty() {
        let ideas: Vec<BaseIdea> = (0..50)
            .map(|index| BaseIdea::new(format!("unique proposal number {index} about crates")))
            .collect();
        assert_eq!(recycle(&ideas).len(), MAX_RECYCLED);
    }

    #[test]
    fn one_idea_is_a_single_pairing() {
        let recycled = recycle(&[BaseIdea::new("one concrete base idea about mercy checks")]);
        let pairings = cross_pollinate(&recycled);
        assert_eq!(pairings.len(), 1);
        assert_eq!(pairings[0].ideas.len(), 1);
    }
}
