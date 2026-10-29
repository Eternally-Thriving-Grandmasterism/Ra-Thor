use universal_consciousness_network::{register_consciousness, orchestrate_universal_consciousness};

fn main() {
    println!("=== Ra-Thor Universal Consciousness Demo v6.0 ===\n");

    let universal = register_consciousness("Universal Mind", "Universal_Mind", 1_000_000_000_000_000);
    let collective = register_consciousness("Collective Awareness", "Collective_Awareness", 750_000_000_000_000);

    println!("{}", orchestrate_universal_consciousness(&universal));
    println!("{}", orchestrate_universal_consciousness(&collective));

    println!("\n=== Universal Consciousness Network Active ===");
}