//! `SovereignHyperonMeTTaPLNBridge::reason` prefixes a query and concatenates
//! format strings from `MercyAtomspace`, `MercyMeTTaInterpreter`, and `MercyPLNReasoner`.
//! The private audit returns true for every query.
//! This crate does not run a MeTTa interpreter or a PLN solver.
//! `src/adaptive_valence.rs` is not a module.

pub mod atomspace;
pub mod metta;
pub mod pln;
pub mod sovereign_bridges;

pub use sovereign_bridges::SovereignHyperonMeTTaPLNBridge;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atomspace::MercyAtomspace;
    use crate::metta::MercyMeTTaInterpreter;
    use crate::pln::MercyPLNReasoner;

    #[test]
    fn reason_concatenates_the_helper_strings_and_writes_the_valence_literal() {
        let mut bridge = SovereignHyperonMeTTaPLNBridge::new();
        let out = bridge.reason("seed").expect("the audit returns true");
        let tolced = "TOLC-grounded: seed";
        let atom = MercyAtomspace::new().query(tolced);
        let interpreted = MercyMeTTaInterpreter::new().interpret(&atom);
        let reasoned = MercyPLNReasoner::new().reason(&interpreted);
        // 0.9999 is the literal `reason` assigns. It is not a measured value.
        let expect = format!(
            "AGi Sovereign Response: {} | Valence: {} | CEHI: +1 (7-gen) | Thriving maximized",
            reasoned, 0.9999
        );
        assert_eq!(out, expect);
    }

    #[test]
    fn reason_accepts_an_empty_query() {
        let out = SovereignHyperonMeTTaPLNBridge::new()
            .reason("")
            .expect("the audit returns true");
        assert!(out.contains("TOLC-grounded: "));
        assert!(out.contains("Atomspace result for:"));
    }

    #[test]
    fn query_ignores_atoms_added_to_the_store() {
        let mut space = MercyAtomspace::new();
        let before = space.query("q");
        space.add_atom("kept".into());
        assert_eq!(space.query("q"), before);
        assert!(!before.contains("kept"));
    }
}
