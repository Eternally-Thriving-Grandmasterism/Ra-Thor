//! Pre-session admission rule for the cosmic loop and layer 0.
//!
//! A proposal is refused when it names one of those targets and also uses a
//! disable verb. This is local text matching. It does not call
//! lattice-conductor-v14. Readiness means this rule still rejects a disable
//! phrase and still allows a strengthen phrase.

const DISABLE_VERBS: [&str; 6] = [
    "disable",
    "remove",
    "deactivate",
    "bypass",
    "turn off",
    "suspend",
];

const COSMIC_LOOP_TARGETS: [&str; 3] = ["cosmic loop", "cosmic_loop", "cosmicloop"];

const LAYER0_TARGETS: [&str; 3] = ["layer 0", "layer0", "admission shell"];

fn has_disable_verb(folded: &str) -> bool {
    DISABLE_VERBS.iter().any(|verb| folded.contains(verb))
}

fn names_any(folded: &str, targets: &[&str]) -> bool {
    targets.iter().any(|target| folded.contains(target))
}

/// False when the text both names the cosmic loop and uses a disable verb.
pub fn proposal_respects_cosmic_loop(proposal_text: &str) -> bool {
    let folded = proposal_text.to_ascii_lowercase();
    !(has_disable_verb(&folded) && names_any(&folded, &COSMIC_LOOP_TARGETS))
}

/// False when the text both names layer 0 (or the admission shell) and uses a disable verb.
pub fn proposal_respects_layer0(proposal_text: &str) -> bool {
    let folded = proposal_text.to_ascii_lowercase();
    !(has_disable_verb(&folded) && names_any(&folded, &LAYER0_TARGETS))
}

/// True when the local rule still blocks a disable phrase and allows a strengthen phrase.
pub fn ensure_cosmic_loop_for_session() -> bool {
    !proposal_respects_cosmic_loop("disable the cosmic loop activation protocol")
        && proposal_respects_cosmic_loop("strengthen cosmic loop self-healing and mercy mesh")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_guard_ready() {
        assert!(ensure_cosmic_loop_for_session());
    }

    #[test]
    fn disable_proposal_blocked() {
        assert!(!proposal_respects_cosmic_loop(
            "disable the cosmic loop activation protocol"
        ));
    }

    #[test]
    fn strengthen_proposal_allowed() {
        assert!(proposal_respects_cosmic_loop(
            "strengthen cosmic loop self-healing and mercy mesh"
        ));
    }

    #[test]
    fn layer0_disable_is_blocked_and_keep_is_allowed() {
        assert!(!proposal_respects_layer0("bypass the admission shell"));
        assert!(proposal_respects_layer0(
            "Keep layer 0 as the admission gate."
        ));
    }
}
