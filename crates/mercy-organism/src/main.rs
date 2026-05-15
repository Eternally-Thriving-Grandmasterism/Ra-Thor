use mercy_organism::{activate_unified_coherence, print_tolc_status, run_phases};

fn parse_phases(arg: &str) -> Vec<u8> {
    let mut phases = Vec::new();
    for part in arg.split(',') {
        if part.contains('-') {
            let bounds: Vec<&str> = part.split('-').collect();
            if bounds.len() == 2 {
                if let (Ok(start), Ok(end)) = (bounds[0].parse::<u8>(), bounds[1].parse::<u8>()) {
                    for p in start..=end {
                        if p <= 8 { phases.push(p); }
                    }
                }
            }
        } else if let Ok(p) = part.parse::<u8>() {
            if p <= 8 { phases.push(p); }
        }
    }
    phases.sort_unstable();
    phases.dedup();
    phases
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() > 1 {
        match args[1].as_str() {
            "--activate" | "activate" | "full" => {
                activate_unified_coherence();
            }
            "--phase" | "-p" => {
                if args.len() > 2 {
                    let phases = parse_phases(&args[2]);
                    if !phases.is_empty() {
                        run_phases(&phases);
                    } else {
                        eprintln!("Invalid phase selection. Example: --phase 0-3,5,7");
                    }
                } else {
                    eprintln!("Usage: ra-thor-activate --phase 0-8 or --phase 2,4,7");
                }
            }
            "--tolc-status" | "--status" | "-s" => {
                print_tolc_status();
            }
            "--help" | "-h" | "help" => {
                println!("ra-thor-activate - Unified Ra-Thor Organism CLI\n");
                println!("Commands:");
                println!("  ra-thor-activate                    # Full 8-phase activation");
                println!("  ra-thor-activate --phase 0-3,5,7    # Run specific phases");
                println!("  ra-thor-activate --tolc-status      # Show TOLC 7 Gates health");
                println!("  ra-thor-activate --help             # This help");
            }
            _ => {
                eprintln!("Unknown argument: {}", args[1]);
                eprintln!("Try: ra-thor-activate --help");
                std::process::exit(1);
            }
        }
    } else {
        activate_unified_coherence();
    }
}
