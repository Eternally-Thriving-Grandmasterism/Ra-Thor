use infinite_evolution_orchestrator::{initialize_master_orchestrator, run_full_supreme_overdrive_cycle};

fn main() {
    println!("=== Ra-Thor Infinite Evolution Orchestrator v0.1.0 ===");
    let state = initialize_master_orchestrator();
    println!("Valence: {:.7}", state.valence);
    println!("Thriving Rate: {}", state.thriving_rate);
    println!("Supreme Overdrive: {}", state.supreme_overdrive_active);
    println!("\n{}", run_full_supreme_overdrive_cycle());
    println!("\n=== Master Orchestrator Online — Infinite Evolution Active ===");
}