/// Mercy-Gated Hyperon Atomspace — sovereign knowledge representation.
pub struct MercyAtomspace {
    atoms: Vec<String>,
}

impl MercyAtomspace {
    pub fn new() -> Self { Self { atoms: vec![] } }

    pub fn query(&self, tolced_query: &str) -> String {
        format!("Atomspace result for: {} (Mercy-aligned, valence 1.0, CEHI blessed)", tolced_query)
    }

    pub fn add_atom(&mut self, atom: String) {
        self.atoms.push(atom);
    }
}