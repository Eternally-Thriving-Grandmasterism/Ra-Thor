//! Innovations generator.
//!
//! Reads base ideas, recycles them, cross-pollinates pairs, and returns at most
//! three ranked seeds. Each seed carries a claim tier and a kill test.
//! Tiers are `idea`, `plausible`, and `evidenced`. A tier records what the
//! input attached. It is not a measurement.
//!
//! Process note:
//! `docs/science/INNOVATIONS-GENERATOR-RUN-2026-09-07-TOLC-BASE-REALITY.md`.

use std::collections::BTreeSet;

use idea_recycling::{cross_pollinate, recycle, BaseIdea, Pairing, RecycledIdea};
use serde::{Deserialize, Serialize};

/// Maximum seeds written for one input.
pub const MAX_SEEDS: usize = 3;

pub use idea_recycling::BaseIdea as InputIdea;

/// Claim tier for one seed.
///
/// - `idea`: no source was attached.
/// - `plausible`: at least one source was attached, and the evidence bar below was not met.
/// - `evidenced`: at least one source was attached, and every idea in the seed carried an evidence note.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ClaimTier {
    Idea,
    Plausible,
    Evidenced,
}

impl ClaimTier {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Idea => "idea",
            Self::Plausible => "plausible",
            Self::Evidenced => "evidenced",
        }
    }
}

/// One ranked seed.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Seed {
    pub rank: u32,
    pub title: String,
    pub summary: String,
    pub claim_tier: ClaimTier,
    pub sources: Vec<String>,
    pub kill_test: String,
}

/// JSON document written by `innovate`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct InnovateReport {
    pub seeds: Vec<Seed>,
}

/// Recycle → cross-pollinate → ranked seeds.
#[derive(Clone, Copy, Debug, Default)]
pub struct InnovationsGenerator;

impl InnovationsGenerator {
    pub fn new() -> Self {
        Self
    }

    pub fn generate_innovations(&self, base_ideas: &[BaseIdea]) -> Vec<Seed> {
        innovate(base_ideas)
    }
}

/// Build at most [`MAX_SEEDS`] seeds. Same input yields the same seeds.
///
/// Higher-scoring pairs are kept first, without reusing a base idea.
/// An idea that was not paired is appended on its own. That stops one long
/// line from occupying every rank.
pub fn innovate(base_ideas: &[BaseIdea]) -> Vec<Seed> {
    let recycled = recycle(base_ideas);
    let pairings = cross_pollinate(&recycled);
    choose_pairings(&recycled, &pairings, MAX_SEEDS)
        .into_iter()
        .enumerate()
        .map(|(index, pairing)| seed_from_pairing(index as u32 + 1, &pairing))
        .collect()
}

fn choose_pairings(recycled: &[RecycledIdea], pairings: &[Pairing], limit: usize) -> Vec<Pairing> {
    if recycled.len() <= 1 {
        return pairings.iter().take(limit).cloned().collect();
    }
    let mut chosen = Vec::new();
    let mut used: BTreeSet<String> = BTreeSet::new();
    for pairing in pairings {
        if chosen.len() == limit {
            break;
        }
        if pairing.ideas.len() < 2 {
            continue;
        }
        if pairing.ideas.iter().any(|idea| used.contains(&idea.text)) {
            continue;
        }
        for idea in &pairing.ideas {
            used.insert(idea.text.clone());
        }
        chosen.push(pairing.clone());
    }
    for idea in recycled {
        if chosen.len() == limit {
            break;
        }
        if used.contains(&idea.text) {
            continue;
        }
        used.insert(idea.text.clone());
        chosen.push(Pairing {
            ideas: vec![idea.clone()],
            shared_themes: idea.themes.clone(),
            score: idea.themes.len() as u64,
        });
    }
    chosen
}

/// Parse base ideas from a JSON array, one JSON object, or `{"base_ideas": ...}`.
pub fn parse_base_ideas(raw: &str) -> Result<Vec<BaseIdea>, String> {
    let document: Document =
        serde_json::from_str(raw).map_err(|err| format!("invalid base ideas JSON: {err}"))?;
    let entries = match document {
        Document::Wrapped(wrapped) => wrapped.base_ideas,
        Document::List(entries) => entries,
        Document::One(entry) => vec![entry],
    };
    Ok(entries.into_iter().map(RawEntry::into_base_idea).collect())
}

pub fn render_report(seeds: &[Seed]) -> Result<String, String> {
    let report = InnovateReport {
        seeds: seeds.to_vec(),
    };
    let mut json = serde_json::to_string_pretty(&report).map_err(|err| err.to_string())?;
    json.push('\n');
    Ok(json)
}

pub fn parse_report(raw: &str) -> Result<InnovateReport, String> {
    serde_json::from_str(raw).map_err(|err| format!("invalid innovate report: {err}"))
}

fn seed_from_pairing(rank: u32, pairing: &Pairing) -> Seed {
    let sources = collect_sources(&pairing.ideas);
    let claim_tier = claim_tier_for(&pairing.ideas, &sources);
    let title = title_for(&pairing.shared_themes, &pairing.ideas);
    let summary = summary_for(&pairing.shared_themes, &pairing.ideas);
    let kill_test = kill_test_for(&pairing.ideas, &sources);
    Seed {
        rank,
        title,
        summary,
        claim_tier,
        sources,
        kill_test,
    }
}

fn claim_tier_for(ideas: &[RecycledIdea], sources: &[String]) -> ClaimTier {
    let sourced = !sources.is_empty();
    let all_notes = !ideas.is_empty()
        && ideas.iter().all(|idea| {
            idea.evidence
                .as_ref()
                .map(|note| !note.is_empty())
                .unwrap_or(false)
        });
    if sourced && all_notes {
        ClaimTier::Evidenced
    } else if sourced {
        ClaimTier::Plausible
    } else {
        ClaimTier::Idea
    }
}

fn collect_sources(ideas: &[RecycledIdea]) -> Vec<String> {
    let mut sources = Vec::new();
    for idea in ideas {
        for source in &idea.sources {
            if !sources.iter().any(|existing| existing == source) {
                sources.push(source.clone());
            }
        }
    }
    sources.sort();
    sources
}

fn title_for(shared: &[String], ideas: &[RecycledIdea]) -> String {
    let body = ideas
        .iter()
        .map(|idea| excerpt(&idea.text, 48))
        .collect::<Vec<_>>()
        .join(" × ");
    match shared.first() {
        Some(theme) => excerpt(&format!("{theme}: {body}"), 140),
        None => excerpt(&body, 140),
    }
}

fn summary_for(shared: &[String], ideas: &[RecycledIdea]) -> String {
    let body = ideas
        .iter()
        .map(|idea| excerpt(&idea.text, 100))
        .collect::<Vec<_>>()
        .join(" | ");
    let theme_label = if ideas.len() == 1 {
        "Themes"
    } else {
        "Shared themes"
    };
    let themes = if shared.is_empty() {
        format!("{theme_label}: none.")
    } else {
        format!("{theme_label}: {}.", shared.join(", "))
    };
    let noun = if ideas.len() == 1 {
        "base idea"
    } else {
        "base ideas"
    };
    format!("Seed from {} {noun}: {body}. {themes}", ideas.len())
}

fn kill_test_for(ideas: &[RecycledIdea], sources: &[String]) -> String {
    let claims = ideas
        .iter()
        .map(|idea| format!("\"{}\"", excerpt(&idea.text, 80)))
        .collect::<Vec<_>>()
        .join(" and ");
    let look = match sources.first() {
        Some(source) => format!("Look first in \"{source}\"."),
        None => {
            "No source was attached, so name one before treating the claim as checked.".to_string()
        }
    };
    format!(
        "Falsify {claims}. {look} Quote a passage that contradicts the claim. If you find that passage, discard this seed."
    )
}

fn excerpt(text: &str, max_chars: usize) -> String {
    let count = text.chars().count();
    if count <= max_chars {
        return text.to_string();
    }
    let keep = max_chars.saturating_sub(3);
    let mut out: String = text.chars().take(keep).collect();
    out.push_str("...");
    out
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Document {
    Wrapped(WrappedIdeas),
    List(Vec<RawEntry>),
    One(RawEntry),
}

#[derive(Deserialize)]
struct WrappedIdeas {
    base_ideas: Vec<RawEntry>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum RawEntry {
    Text(String),
    Object(RawObject),
}

#[derive(Deserialize)]
struct RawObject {
    #[serde(default)]
    text: String,
    #[serde(default)]
    summary: String,
    #[serde(default)]
    title: String,
    #[serde(default)]
    sources: Vec<String>,
    #[serde(default)]
    evidence: Option<String>,
}

impl RawEntry {
    fn into_base_idea(self) -> BaseIdea {
        match self {
            Self::Text(text) => BaseIdea::new(text),
            Self::Object(object) => {
                let text = [object.text, object.summary, object.title]
                    .into_iter()
                    .find(|candidate| !candidate.trim().is_empty())
                    .unwrap_or_default();
                BaseIdea {
                    text,
                    sources: object.sources,
                    evidence: object.evidence,
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn many_ideas() -> Vec<BaseIdea> {
        (0..6)
            .map(|index| {
                BaseIdea::new(format!(
                    "distinct proposal {index} about crate workflow number {index}"
                ))
                .with_sources(vec![format!("notes/{index}.md")])
            })
            .collect()
    }

    #[test]
    fn five_ideas_do_not_repeat_one_line_in_every_seed() {
        let ideas = vec![
            BaseIdea::new("Displacement is a public-service concern")
                .with_sources(vec!["notes/layer-a.md".into()]),
            BaseIdea::new("Soft feedback events are sealed telemetry")
                .with_sources(vec!["notes/layer-b.md".into()]),
            BaseIdea::new("Abundance claims need a cash path")
                .with_sources(vec!["docs/REVENUE_SCALE.md".into()]),
            BaseIdea::new("Proof ladder is a queue for surmise"),
            BaseIdea::new("A research label stays a surmise until a named check")
                .with_sources(vec!["docs/science/minute.md".into()]),
        ];
        let seeds = innovate(&ideas);
        assert_eq!(seeds.len(), 3);
        assert!(seeds[0].summary.to_lowercase().contains("surmise"));
        let repeated = seeds
            .iter()
            .filter(|seed| seed.summary.contains("research label"))
            .count();
        assert!(repeated <= 1);
        assert_eq!(innovate(&ideas), seeds);
    }

    #[test]
    fn caps_at_three_ranked_seeds() {
        let seeds = innovate(&many_ideas());
        assert_eq!(seeds.len(), MAX_SEEDS);
        assert_eq!(
            seeds.iter().map(|seed| seed.rank).collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
    }

    #[test]
    fn tiers_are_closed_and_kill_tests_are_present() {
        let mut ideas = many_ideas();
        ideas[0].sources.clear();
        ideas[0].evidence = None;
        let seeds = innovate(&ideas);
        for seed in &seeds {
            assert!(matches!(
                seed.claim_tier,
                ClaimTier::Idea | ClaimTier::Plausible | ClaimTier::Evidenced
            ));
            assert_eq!(
                seed.claim_tier.as_str(),
                serde_json::to_value(seed.claim_tier)
                    .unwrap()
                    .as_str()
                    .unwrap()
            );
            assert!(!seed.kill_test.trim().is_empty());
            assert!(!seed.title.trim().is_empty() || !seed.summary.trim().is_empty());
            let blob = serde_json::to_string(seed).unwrap();
            assert!(!blob.contains("0.999999"));
            assert!(!blob.contains("valence"));
        }
    }

    #[test]
    fn same_input_yields_the_same_seeds() {
        let ideas = many_ideas();
        assert_eq!(innovate(&ideas), innovate(&ideas));
        let mut reversed = ideas.clone();
        reversed.reverse();
        assert_eq!(innovate(&ideas), innovate(&reversed));
    }

    #[test]
    fn tier_ladder_uses_sources_and_evidence_notes() {
        let bare = BaseIdea::new("alpha displacement concern in public service work");
        let sourced = BaseIdea::new("beta telemetry seal for feedback events")
            .with_sources(vec!["layer-b.md".into()]);
        let evidenced = BaseIdea::new("gamma revenue path for this year cash")
            .with_sources(vec!["REVENUE_SCALE.md".into()])
            .with_evidence("a named page lists the path");

        let bare_seeds = innovate(&[bare]);
        assert_eq!(bare_seeds[0].claim_tier, ClaimTier::Idea);
        assert!(bare_seeds[0].sources.is_empty());

        let sourced_seeds = innovate(std::slice::from_ref(&sourced));
        assert_eq!(sourced_seeds[0].claim_tier, ClaimTier::Plausible);
        assert_eq!(sourced_seeds[0].sources, vec!["layer-b.md".to_string()]);

        let evidenced_seeds = innovate(std::slice::from_ref(&evidenced));
        assert_eq!(evidenced_seeds[0].claim_tier, ClaimTier::Evidenced);

        let mixed = innovate(&[sourced, evidenced.clone()]);
        assert_eq!(mixed[0].claim_tier, ClaimTier::Plausible);
        assert_eq!(
            mixed[0].sources,
            vec!["REVENUE_SCALE.md".to_string(), "layer-b.md".to_string()]
        );
        assert!(mixed[0].kill_test.contains("REVENUE_SCALE.md"));

        let evidenced_too = BaseIdea::new("delta proof ladder is a queue ticket")
            .with_sources(vec!["minute.md".into()])
            .with_evidence("the minute calls surmise a queue ticket");
        let both = innovate(&[evidenced, evidenced_too]);
        assert_eq!(both[0].claim_tier, ClaimTier::Evidenced);
        assert!(!both[0].kill_test.trim().is_empty());
    }

    #[test]
    fn sources_are_only_those_on_the_input() {
        let seeds = innovate(&many_ideas());
        for seed in seeds {
            assert!(seed
                .sources
                .iter()
                .all(|source| source.starts_with("notes/") && source.ends_with(".md")));
        }
    }

    #[test]
    fn parses_strings_objects_and_wrapper() {
        let from_strings = parse_base_ideas(r#"["one idea about mercy checks", "two"]"#).unwrap();
        assert_eq!(from_strings.len(), 2);
        assert_eq!(from_strings[0].text, "one idea about mercy checks");

        let from_objects = parse_base_ideas(
            r#"[{"summary":"paired claim","sources":["a.md"],"evidence":"note"}]"#,
        )
        .unwrap();
        assert_eq!(from_objects[0].text, "paired claim");
        assert_eq!(from_objects[0].sources, vec!["a.md".to_string()]);
        assert_eq!(from_objects[0].evidence.as_deref(), Some("note"));

        let wrapped = parse_base_ideas(r#"{"base_ideas":[{"title":"from title"}]}"#).unwrap();
        assert_eq!(wrapped[0].text, "from title");
    }

    #[test]
    fn unknown_tier_does_not_parse() {
        let raw = r#"{"seeds":[{"rank":1,"title":"t","summary":"s","claim_tier":"fact","sources":[],"kill_test":"k"}]}"#;
        assert!(parse_report(raw).is_err());
    }

    #[test]
    fn empty_input_has_no_seeds() {
        assert!(innovate(&[]).is_empty());
        assert!(parse_base_ideas("[]").unwrap().is_empty());
    }
}
