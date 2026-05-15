use mercy_organism::{activate_unified_coherence, print_tolc_status, run_phases, get_activation_result_json, get_activation_result_compact_json, get_activation_result_yaml, get_activation_result_toml, get_prometheus_metrics, load_config, start_grpc_server, start_websocket_server, auto_heal, activate_patsagi_councils, activate_powrush_rbe, start_database_backend, auto_scale, activate_interstellar_operations, start_kubernetes_operator, start_prometheus_exporter};

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

    let config = load_config("ra-thor-activate.toml");
    let default_phases = config.as_ref().and_then(|c| c.default_phases.clone()).unwrap_or_else(|| (0..=8).collect());

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
            "--json" | "-j" => {
                let phases: Vec<u8> = if args.len() > 2 {
                    parse_phases(&args[2])
                } else {
                    default_phases.clone()
                };
                println!("{}", get_activation_result_json(&phases));
            }
            "--json-compact" | "--jsonc" | "-jc" => {
                let phases: Vec<u8> = if args.len() > 2 {
                    parse_phases(&args[2])
                } else {
                    default_phases.clone()
                };
                println!("{}", get_activation_result_compact_json(&phases));
            }
            "--yaml" | "-y" => {
                let phases: Vec<u8> = if args.len() > 2 {
                    parse_phases(&args[2])
                } else {
                    default_phases.clone()
                };
                println!("{}", get_activation_result_yaml(&phases));
            }
            "--toml" | "-t" => {
                let phases: Vec<u8> = if args.len() > 2 {
                    parse_phases(&args[2])
                } else {
                    default_phases.clone()
                };
                println!("{}", get_activation_result_toml(&phases));
            }
            "--prometheus" | "--metrics" | "-m" => {
                let phases: Vec<u8> = if args.len() > 2 {
                    parse_phases(&args[2])
                } else {
                    default_phases.clone()
                };
                println!("{}", get_prometheus_metrics(&phases));
            }
            "--config" => {
                if args.len() > 2 {
                    if let Some(cfg) = load_config(&args[2]) {
                        println!("Loaded config: {:?}", cfg);
                    } else {
                        eprintln!("Failed to load config file");
                    }
                } else {
                    println!("Usage: ra-thor-activate --config ra-thor-activate.toml");
                }
            }
            "--serve" | "--grpc" => {
                start_grpc_server();
            }
            "--websocket" => {
                start_websocket_server();
            }
            "--heal" | "--auto-heal" => {
                auto_heal();
            }
            "--db" | "--database" => {
                start_database_backend();
            }
            "--scale" | "--auto-scale" => {
                auto_scale();
            }
            "--k8s" | "--kubernetes" => {
                start_kubernetes_operator();
            }
            "--exporter" | "--prometheus-exporter" => {
                start_prometheus_exporter();
            }
            "--help" | "-h" | "help" => {
                println!("ra-thor-activate - Unified Ra-Thor Organism CLI\n");
                println!("Commands:");
                println!("  ra-thor-activate                           # Full 8-phase activation");
                println!("  ra-thor-activate --phase 0-3,5,7           # Run specific phases");
                println!("  ra-thor-activate --tolc-status             # Show TOLC 7 Gates health");
                println!("  ra-thor-activate --json [phases]          # Pretty JSON");
                println!("  ra-thor-activate --json-compact [phases]  # Compact JSON");
                println!("  ra-thor-activate --yaml [phases]          # YAML");
                println!("  ra-thor-activate --toml [phases]          # TOML");
                println!("  ra-thor-activate --prometheus [phases]    # Prometheus metrics");
                println!("  ra-thor-activate --config <file>          # Load config");
                println!("  ra-thor-activate --serve / --grpc          # Start gRPC endpoint");
                println!("  ra-thor-activate --websocket              # Start WebSocket server");
                println!("  ra-thor-activate --heal                   # Auto-healing self-repair");
                println!("  ra-thor-activate --db / --database        # Start database backend");
                println!("  ra-thor-activate --scale / --auto-scale   # Auto-scaling");
                println!("  ra-thor-activate --k8s / --kubernetes     # Start Kubernetes operator");
                println!("  ra-thor-activate --exporter               # Start Prometheus exporter");
                println!("  ra-thor-activate --help                   # This help");
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
