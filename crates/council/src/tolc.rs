//! Token overlap for one member.
//!
//! [`TOLC_8`] lists the public names. Resonance here is the fraction of a
//! member's tokens that appear in the proposal. It is not a gate execution
//! and not a valence.

/// Public TOLC 8 names.
pub const TOLC_8: [&str; 8] = [
    "Truth",
    "Order",
    "Love",
    "Compassion",
    "Service",
    "Abundance",
    "Joy",
    "Cosmic Harmony",
];

/// Fraction of `tokens` that appear in `text`.
///
/// Matching is ASCII case-insensitive substring search.
/// An empty token list returns 0.
pub fn calculate_tolc_resonance(tokens: &[&str], text: &str) -> f64 {
    if tokens.is_empty() {
        return 0.0;
    }
    let folded = text.to_ascii_lowercase();
    let hits = tokens
        .iter()
        .filter(|token| folded.contains(*token))
        .count();
    hits as f64 / tokens.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_overlap_is_one_and_none_is_zero() {
        let tokens = ["truth", "record", "human"];
        assert_eq!(
            calculate_tolc_resonance(&tokens, "Record the human truth"),
            1.0
        );
        assert_eq!(calculate_tolc_resonance(&tokens, "paint the bikeshed"), 0.0);
        assert_eq!(calculate_tolc_resonance(&tokens, "record only"), 1.0 / 3.0);
        assert_eq!(calculate_tolc_resonance(&[], "record the human truth"), 0.0);
        assert_ne!(
            calculate_tolc_resonance(&tokens, "Record the human truth"),
            0.9999999
        );
    }

    #[test]
    fn public_names_are_the_eight() {
        assert_eq!(TOLC_8.len(), 8);
        assert_eq!(TOLC_8[0], "Truth");
        assert_eq!(TOLC_8[7], "Cosmic Harmony");
    }
}
