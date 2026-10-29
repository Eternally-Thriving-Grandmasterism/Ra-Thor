/// Formats `interpreted` with a fixed prefix. Not a PLN solver.
/// The words `0.999 probability` in that prefix are source text, not a measured probability.
pub struct MercyPLNReasoner;

impl MercyPLNReasoner {
    pub fn new() -> Self {
        Self
    }

    pub fn reason(&self, interpreted: &str) -> String {
        format!("PLN reasoned outcome (truth + mercy + thriving + 0.999 probability): {} | CEHI eternal", interpreted)
    }
}

impl Default for MercyPLNReasoner {
    fn default() -> Self {
        Self::new()
    }
}
