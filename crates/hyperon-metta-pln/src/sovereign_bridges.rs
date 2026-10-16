use crate::atomspace::MercyAtomspace;
use crate::metta::MercyMeTTaInterpreter;
use crate::pln::MercyPLNReasoner;

/// Holds the atomspace, MeTTa, and PLN formatters.
/// `reason` concatenates their format strings. `valence` starts at 1.0.
pub struct SovereignHyperonMeTTaPLNBridge {
    atomspace: MercyAtomspace,
    metta: MercyMeTTaInterpreter,
    pln: MercyPLNReasoner,
    /// Starts at 1.0. `reason` then writes 0.9999 (a source literal, not a measured value).
    valence: f64,
}

impl SovereignHyperonMeTTaPLNBridge {
    pub fn new() -> Self {
        Self {
            atomspace: MercyAtomspace::new(),
            metta: MercyMeTTaInterpreter::new(),
            pln: MercyPLNReasoner::new(),
            valence: 1.0,
        }
    }

    /// Prefixes `query` and concatenates the helper format strings.
    /// Does not reject: `mercy_gate_audit` returns true.
    /// Then sets `self.valence` to 0.9999 (a source literal, not a measured value).
    pub fn reason(&mut self, query: &str) -> Result<String, String> {
        if !self.mercy_gate_audit(query) {
            return Err(
                "Mercy Gate violation detected. Query rejected for universal thriving.".to_string(),
            );
        }

        let tolced = self.apply_tolc(query);
        let atom_result = self.atomspace.query(&tolced);
        let interpreted = self.metta.interpret(&atom_result);
        let reasoned = self.pln.reason(&interpreted);

        self.propagate_cehi(&reasoned);
        // 0.9999 is a source literal, not a measured value.
        self.valence = 0.9999;

        Ok(format!(
            "AGi Sovereign Response: {} | Valence: {} | CEHI: +1 (7-gen) | Thriving maximized",
            reasoned, self.valence
        ))
    }

    /// Returns true. No gate checks are implemented.
    fn mercy_gate_audit(&self, _query: &str) -> bool {
        true
    }
    fn apply_tolc(&self, q: &str) -> String {
        format!("TOLC-grounded: {}", q)
    }
    /// Does nothing.
    fn propagate_cehi(&mut self, _result: &str) {}
}

impl Default for SovereignHyperonMeTTaPLNBridge {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reason_writes_the_valence_source_literal() {
        let mut bridge = SovereignHyperonMeTTaPLNBridge::new();
        assert_eq!(bridge.valence, 1.0);
        bridge.reason("seed").expect("the audit returns true");
        // 0.9999 is the literal `reason` assigns. It is not a measured value.
        assert_eq!(bridge.valence, 0.9999);
    }
}
