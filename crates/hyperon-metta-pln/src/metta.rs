/// Formats `atom_result` with a fixed prefix. Not a MeTTa interpreter.
pub struct MercyMeTTaInterpreter;

impl MercyMeTTaInterpreter {
    pub fn new() -> Self {
        Self
    }

    pub fn interpret(&self, atom_result: &str) -> String {
        format!(
            "MeTTa interpreted (sovereign, joyful, abundant, TOLC-aligned): {}",
            atom_result
        )
    }
}

impl Default for MercyMeTTaInterpreter {
    fn default() -> Self {
        Self::new()
    }
}
