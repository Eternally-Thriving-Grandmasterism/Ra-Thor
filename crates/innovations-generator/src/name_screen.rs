//! Deterministic character-name Mercy-Checks (heuristic, English word lists).
//!
//! Does not clear trademarks, repo clashes, or non-English meanings.

const COMMON_NOUNS_RAW: &str = include_str!("../data/common_nouns.txt");
const NEGATIVE_WORDS_RAW: &str = include_str!("../data/negative_tease_words.txt");

const NEEDS_RESEARCH: &str =
    "NEEDS RESEARCH: IP/trademark and own-repo clash search";
const NEEDS_NATIVE: &str = "NEEDS NATIVE CHECK: languages not checked";
const NEEDS_HUMAN: &str =
    "NEEDS HUMAN: associations (ethnic, religious, place) and harmonic feel";
const NOT_LEGAL_TM: &str = "Not legal trademark clearance.";

/// Outcome of one Mercy-Check line on a name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LineKind {
    Pass,
    Fail,
    Flag,
}

/// One line in a per-name report.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReportLine {
    pub kind: LineKind,
    pub message: String,
}

/// Full screen result for one candidate name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NameScreenResult {
    pub name: String,
    pub lines: Vec<ReportLine>,
}

impl NameScreenResult {
    pub fn hard_failed(&self) -> bool {
        self.lines
            .iter()
            .any(|line| line.kind == LineKind::Fail)
    }
}

/// Screen one name against optional cast members (other names already in the cast).
pub fn screen_name(name: &str, cast: &[String]) -> NameScreenResult {
    let trimmed = name.trim();
    let display = if trimmed.is_empty() {
        name.to_string()
    } else {
        trimmed.to_string()
    };
    let lower = display.to_lowercase();

    let mut lines = Vec::new();

    // Character-ness (hard)
    if check_character_ness(&lower, &display) {
        lines.push(ReportLine {
            kind: LineKind::Pass,
            message: "Character-ness: PASS".into(),
        });
    } else {
        lines.push(ReportLine {
            kind: LineKind::Fail,
            message: "Character-ness: FAIL (matches bundled common noun/material list)"
                .into(),
        });
    }

    // Sound-alikes (hard)
    match check_sound_alikes(&lower) {
        None => {
            lines.push(ReportLine {
                kind: LineKind::Pass,
                message: "Sound-alikes: PASS".into(),
            });
        }
        Some(trigger) => {
            lines.push(ReportLine {
                kind: LineKind::Fail,
                message: format!("Sound-alikes: FAIL (trigger: {trigger})"),
            });
        }
    }

    // Shape (soft)
    let syllables = estimate_syllables(&lower);
    if syllables == 1 || syllables == 2 {
        lines.push(ReportLine {
            kind: LineKind::Pass,
            message: format!("Shape: PASS (approx {syllables} syllable(s))"),
        });
    } else {
        lines.push(ReportLine {
            kind: LineKind::Flag,
            message: format!(
                "Shape: FLAG (approx {syllables} syllable(s); heuristic expects 1–2)"
            ),
        });
    }

    // Cast clash (soft)
    let tail = rhyme_tail(&lower);
    let first = lower.chars().next();
    let mut cast_flags: Vec<String> = Vec::new();
    for other in cast {
        let other_trim = other.trim();
        if other_trim.is_empty() || other_trim.eq_ignore_ascii_case(&display) {
            continue;
        }
        let other_lower = other_trim.to_lowercase();
        if first.is_some() && other_lower.chars().next() == first {
            cast_flags.push(format!("shared first letter with \"{other_trim}\""));
        }
        if !tail.is_empty() && tail == rhyme_tail(&other_lower) {
            cast_flags.push(format!(
                "identical rhyme tail \"{tail}\" with \"{other_trim}\""
            ));
        }
    }
    if cast_flags.is_empty() {
        lines.push(ReportLine {
            kind: LineKind::Pass,
            message: "Cast clash: PASS".into(),
        });
    } else {
        lines.push(ReportLine {
            kind: LineKind::Flag,
            message: format!("Cast clash: FLAG ({})", cast_flags.join("; ")),
        });
    }

    for fixed in [NEEDS_RESEARCH, NEEDS_NATIVE, NEEDS_HUMAN, NOT_LEGAL_TM] {
        lines.push(ReportLine {
            kind: LineKind::Pass,
            message: fixed.into(),
        });
    }

    NameScreenResult {
        name: display,
        lines,
    }
}

/// Render one name's report as plain text lines.
pub fn render_name_report(result: &NameScreenResult) -> String {
    let mut out = format!("=== {} ===\n", result.name);
    for line in &result.lines {
        if is_fixed_needs_line(&line.message) {
            out.push_str(&format!("{}\n", line.message));
            continue;
        }
        let tag = match line.kind {
            LineKind::Pass => "PASS",
            LineKind::Fail => "FAIL",
            LineKind::Flag => "FLAG",
        };
        out.push_str(&format!("{tag}: {}\n", line.message));
    }
    out
}

/// Screen many names; returns results and whether any hard check failed.
pub fn screen_names(names: &[String], cast: &[String]) -> (Vec<NameScreenResult>, bool) {
    let mut results = Vec::new();
    let mut any_fail = false;
    for name in names {
        let result = screen_name(name, cast);
        if result.hard_failed() {
            any_fail = true;
        }
        results.push(result);
    }
    (results, any_fail)
}

fn load_word_list(raw: &str) -> Vec<String> {
    raw.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(|line| line.to_lowercase())
        .collect()
}

fn common_nouns() -> Vec<String> {
    load_word_list(COMMON_NOUNS_RAW)
}

fn negative_words() -> Vec<String> {
    load_word_list(NEGATIVE_WORDS_RAW)
}

fn has_name_like_ending(lower: &str) -> bool {
    lower.ends_with('y') || lower.ends_with("ie") || lower.ends_with('o')
}

fn check_character_ness(lower: &str, _display: &str) -> bool {
    if lower.is_empty() {
        return true;
    }
    if has_name_like_ending(lower) {
        return true;
    }
    !common_nouns().iter().any(|noun| noun == lower)
}

fn check_sound_alikes(lower: &str) -> Option<String> {
    if lower.is_empty() {
        return None;
    }
    let tail = rhyme_tail(lower);
    for word in negative_words() {
        if lower == word {
            return Some(word);
        }
        let word_tail = rhyme_tail(&word);
        if tail.len() >= 3 && !tail.is_empty() && tail == word_tail {
            return Some(word);
        }
        let syllable = first_syllable(&word);
        if syllable.len() >= 2 && lower.starts_with(&syllable) {
            return Some(word);
        }
    }
    None
}

/// Last vowel group plus following consonants (English letters only).
pub fn rhyme_tail(lower: &str) -> String {
    let letters: Vec<char> = lower.chars().filter(|c| c.is_ascii_alphabetic()).collect();
    if letters.is_empty() {
        return String::new();
    }
    let mut last_group_start: Option<usize> = None;
    for (index, ch) in letters.iter().enumerate() {
        if is_vowel(*ch) {
            let group_start = if index > 0 && is_vowel(letters[index - 1]) {
                last_group_start.unwrap_or(index)
            } else {
                index
            };
            last_group_start = Some(group_start);
        }
    }
    match last_group_start {
        None => String::new(),
        Some(start) => letters[start..].iter().collect(),
    }
}

fn is_vowel(ch: char) -> bool {
    matches!(ch, 'a' | 'e' | 'i' | 'o' | 'u' | 'y')
}

/// Onset + nucleus + coda consonants before the next vowel (approximate first syllable).
pub fn first_syllable(lower: &str) -> String {
    let letters: Vec<char> = lower.chars().filter(|c| c.is_ascii_alphabetic()).collect();
    if letters.is_empty() {
        return String::new();
    }
    let mut index = 0;
    while index < letters.len() && !is_vowel(letters[index]) {
        index += 1;
    }
    if index >= letters.len() {
        return letters.iter().collect();
    }
    while index < letters.len() && is_vowel(letters[index]) {
        index += 1;
    }
    if index < letters.len() && !is_vowel(letters[index]) {
        index += 1;
    }
    letters[..index].iter().collect()
}

/// Approximate syllable count via vowel-group heuristic (English letters only).
pub fn estimate_syllables(lower: &str) -> usize {
    let letters: Vec<char> = lower.chars().filter(|c| c.is_ascii_alphabetic()).collect();
    if letters.is_empty() {
        return 0;
    }
    let mut count = 0;
    let mut in_vowel = false;
    for ch in letters {
        if is_vowel(ch) {
            if !in_vowel {
                count += 1;
                in_vowel = true;
            }
        } else {
            in_vowel = false;
        }
    }
    count.max(1)
}

fn is_fixed_needs_line(message: &str) -> bool {
    message.starts_with("NEEDS ") || message == NOT_LEGAL_TM
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hard_lines(result: &NameScreenResult) -> Vec<&ReportLine> {
        result
            .lines
            .iter()
            .filter(|line| !is_fixed_needs_line(&line.message))
            .collect()
    }

    fn rendered(result: &NameScreenResult) -> String {
        render_name_report(result)
    }

    #[test]
    fn coil_fails_common_noun() {
        let result = screen_name("Coil", &[]);
        assert!(result.hard_failed());
        assert!(
            hard_lines(&result)
                .iter()
                .any(|line| line.kind == LineKind::Fail && line.message.contains("Character-ness"))
        );
    }

    #[test]
    fn coily_passes_character_ness() {
        let result = screen_name("Coily", &[]);
        let lines = hard_lines(&result);
        let char_line = lines
            .iter()
            .find(|line| line.message.starts_with("Character-ness"))
            .expect("character-ness line");
        assert_eq!(char_line.kind, LineKind::Pass);
    }

    #[test]
    fn spool_fails_rhyme_with_fool() {
        let result = screen_name("Spool", &[]);
        assert!(result.hard_failed());
        let lines = hard_lines(&result);
        let sound = lines
            .iter()
            .find(|line| line.message.starts_with("Sound-alikes"))
            .expect("sound line");
        assert_eq!(sound.kind, LineKind::Fail);
        assert!(sound.message.contains("fool"));
    }

    #[test]
    fn dunstan_fails_dunce_prefix() {
        let result = screen_name("Dunstan", &[]);
        assert!(result.hard_failed());
        let lines = hard_lines(&result);
        let sound = lines
            .iter()
            .find(|line| line.message.starts_with("Sound-alikes"))
            .expect("sound line");
        assert!(sound.message.contains("dunce"));
    }

    #[test]
    fn hesper_passes_hard_checks() {
        let result = screen_name("Hesper", &[]);
        assert!(!result.hard_failed());
        for line in hard_lines(&result) {
            if line.message.starts_with("Character-ness")
                || line.message.starts_with("Sound-alikes")
            {
                assert_eq!(line.kind, LineKind::Pass);
            }
        }
    }

    #[test]
    fn long_name_flags_shape() {
        let result = screen_name("Valentina", &[]);
        let lines = hard_lines(&result);
        let shape = lines
            .iter()
            .find(|line| line.message.starts_with("Shape"))
            .expect("shape line");
        assert_eq!(shape.kind, LineKind::Flag);
    }

    #[test]
    fn needs_lines_always_present() {
        for name in ["Coil", "Hesper", "X"] {
            let text = rendered(&screen_name(name, &[]));
            assert!(text.contains(NEEDS_RESEARCH));
            assert!(text.contains(NEEDS_NATIVE));
            assert!(text.contains(NEEDS_HUMAN));
            assert!(text.contains(NOT_LEGAL_TM));
        }
    }

    #[test]
    fn rhyme_tail_spool_matches_fool() {
        assert_eq!(rhyme_tail("spool"), rhyme_tail("fool"));
    }

    #[test]
    fn first_syllable_dunce_is_dun() {
        assert_eq!(first_syllable("dunce"), "dun");
    }

    #[test]
    fn hesper_does_not_rhyme_loser() {
        assert!(rhyme_tail("hesper").len() < 3);
        let result = screen_name("Hesper", &[]);
        assert!(!result.hard_failed());
    }
}
