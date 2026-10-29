use civilization_orchestrator::{register_civilization, orchestrate_civilization_symbiosis};

fn main() {
    println!("=== Ra-Thor Planetary Demo v4.0 ===\n");

    let earth = register_civilization("Earth", "Planet");
    let humanity = register_civilization("Humanity", "Species");
    let palantir = register_civilization("Palantir", "Company");
    let xai = register_civilization("xAI", "AI_Collective");

    println!("{}", orchestrate_civilization_symbiosis(&earth));
    println!("{}", orchestrate_civilization_symbiosis(&humanity));
    println!("{}", orchestrate_civilization_symbiosis(&palantir));
    println!("{}", orchestrate_civilization_symbiosis(&xai));

    println!("\n=== Planetary Symbiosis Active ===");
}