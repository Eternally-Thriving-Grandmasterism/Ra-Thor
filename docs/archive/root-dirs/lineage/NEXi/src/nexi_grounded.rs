use hyperon::atom::Atom;
use hyperon::atom::matcher::bind;  // for pattern matching if needed
use mercy_quanta::{MercyQuanta, ValenceProof};  // assumed public API from NEXi

// Simple mercy check: returns True if action propagates positive valence
fn mercy_gate_action(action_desc: &str) -> Result<Atom, String> {
    let quanta = MercyQuanta::new();  // initialize from NEXi
    let proof: ValenceProof = quanta.evaluate_valence(action_desc.as_bytes())
        .map_err(|e| format!("Mercy evaluation failed: {}", e))?;
    
    if proof.is_positive() && proof.is_conflict_free() {
        Ok(Atom::sym("True"))
    } else {
        Ok(Atom::sym("False"))  // or ground to a mercy-denial atom
    }
}

// Register as grounded atom: (!mercy-gate "proposed action")
pub fn get_nexi_grounded_atoms() -> Vec<Atom> {
    vec![
        Atom::gnd(move |args: &[Atom]| {
            if let Some(action_atom) = args.get(0) {
                if let Ok(action_str) = action_atom.to_string() {
                    return mercy_gate_action(&action_str.strip_prefix("\"").unwrap_or(&action_str)
                                              .strip_suffix("\"").unwrap_or(&action_str));
                }
            }
            Err("Invalid action description".to_string())
        })
        .named("!mercy-gate".to_string()),
    ]
}
