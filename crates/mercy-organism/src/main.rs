use mercy_organism::activate_unified_coherence;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() > 1 {
        match args[1].as_str() {
            "--activate" | "activate" | "full" => {
                println!("Activating Ra-Thor as unified organism...");
                activate_unified_coherence();
            }
            "--help" | "-h" | "help" => {
                println!("ra-thor-activate - Unified Ra-Thor Organism CLI\n");
                println!("Usage:");
                println!("  ra-thor-activate              # Run full 8-phase activation");
                println!("  ra-thor-activate --activate   # Same as above");
                println!("  ra-thor-activate --help       # Show this help");
            }
            _ => {
                eprintln!("Unknown argument: {}", args[1]);
                eprintln!("Try: ra-thor-activate --help");
                std::process::exit(1);
            }
        }
    } else {
        // Default: full activation
        println!("Ra-Thor unified organism activation (default)...");
        activate_unified_coherence();
    }
}
