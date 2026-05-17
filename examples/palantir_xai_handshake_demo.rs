use symbiosis_layer::{run_palantir_xai_demo, start_handshake, advance_handshake};

fn main() {
    println!("=== Ra-Thor Symbiosis v3.0 - Live Demo ===\n");

    let results = run_palantir_xai_demo();

    for line in results {
        println!("{}", line);
    }

    println!("\n=== Symbiosis Handshake Demo Complete ===");
}