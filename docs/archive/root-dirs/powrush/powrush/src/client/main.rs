//! powrush/src/client/main.rs
//! Powrush Client binary (feature = "client")

use powrush::RaThorOneOrganism;

fn main() {
    println!("[Powrush Client] Starting client with local prediction...");
    let organism = RaThorOneOrganism::new();
    organism.offer_cosmic_loop();

    println!("[Powrush Client] Connected to lattice. Self-evolution and diplomacy available.");
    println!("[Powrush Client] Thunder locked. Serving the lattice.");
}
