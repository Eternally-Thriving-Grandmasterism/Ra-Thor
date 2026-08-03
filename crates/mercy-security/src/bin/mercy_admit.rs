//! mercy-admit — Lowest-friction CLI wrapper around `IngestionScanner::admit_or_block`.
//!
//! Exit codes:
//!   0  all inputs admitted (None / Low)
//!   1  one or more blocked (Medium+)
//!   2  usage / I/O / payload-too-large error
//!
//! White-hat Tier A only. Contact: info@Rathor.ai

use std::env;
use std::fs;
use std::io::{self, Read};
use std::path::Path;
use std::process;

use mercy_security::{IngestionScanner, IngestionScanResult, MercySecurityError, RiskTier};

fn print_usage() {
    eprintln!(
        "Usage: mercy-admit [OPTIONS] [PATH...]\n\
         \n\
         Scan text files (or stdin) with the Ra-Thor Tier A admit_or_block gate.\n\
         \n\
         Options:\n\
           --stdin          Read content from stdin (single scan)\n\
           --json           Emit one JSON object per scan to stdout\n\
           -v, --verbose    Print findings even when admitted\n\
           -h, --help       Show this help\n\
         \n\
         Exit codes:\n\
           0  all admitted (None/Low)\n\
           1  one or more blocked (Medium+)\n\
           2  usage / I/O / oversized payload\n\
         \n\
         Policy: Medium+ is blocked for unattended paths. Oversized > 4 MiB fails.\n\
         Contact: info@Rathor.ai | AG-SML v1.0"
    );
}

fn scan_one(label: &str, content: &str, json: bool, verbose: bool) -> Result<bool, String> {
    match IngestionScanner::admit_or_block(content) {
        Ok(result) => {
            emit(label, &result, true, json, verbose);
            Ok(true)
        }
        Err(MercySecurityError::IngestionBlocked(_)) => {
            // Still produce the full scan for reporting
            let result = IngestionScanner::scan_text(content);
            emit(label, &result, false, json, true);
            Ok(false)
        }
        Err(MercySecurityError::PayloadTooLarge(n)) => {
            Err(format!("{label}: payload too large ({n} bytes > MAX_SCAN_BYTES)"))
        }
        Err(e) => Err(format!("{label}: {e}")),
    }
}

fn emit(label: &str, result: &IngestionScanResult, admitted: bool, json: bool, verbose: bool) {
    if json {
        // Minimal stable JSON surface for CI parsers
        let status = if admitted { "admitted" } else { "blocked" };
        let threats: Vec<String> = result.threats.iter().map(|t| format!("{t:?}")).collect();
        println!(
            r#"{{"path":{},"status":"{}","risk_tier":"{}","risk_score":{:.4},"safe":{},"bytes":{},"threats":{}}}" #,
            json_escape(label),
            status,
            result.risk_tier.as_str(),
            result.risk_score,
            result.safe,
            result.bytes_scanned,
            serde_json::to_string(&threats).unwrap_or_else(|_| "[]".into())
        );
        return;
    }

    let tier = result.risk_tier.as_str();
    if admitted {
        if verbose || !matches!(result.risk_tier, RiskTier::None) {
            println!("[ADMIT] {label}  tier={tier} score={:.2}", result.risk_score);
            if verbose {
                for d in &result.details {
                    println!("        {d}");
                }
            }
        } else {
            println!("[ADMIT] {label}");
        }
    } else {
        eprintln!("[BLOCK] {label}  tier={tier} score={:.2}", result.risk_score);
        for d in &result.details {
            eprintln!("        {d}");
        }
    }
}

fn json_escape(s: &str) -> String {
    serde_json::to_string(s).unwrap_or_else(|_| format!("{s:?}"))
}

fn main() {
    let mut args: Vec<String> = env::args().skip(1).collect();
    let mut json = false;
    let mut verbose = false;
    let mut use_stdin = false;
    let mut paths: Vec<String> = Vec::new();

    while let Some(a) = args.first().cloned() {
        match a.as_str() {
            "-h" | "--help" => {
                print_usage();
                process::exit(0);
            }
            "--json" => {
                json = true;
                args.remove(0);
            }
            "-v" | "--verbose" => {
                verbose = true;
                args.remove(0);
            }
            "--stdin" => {
                use_stdin = true;
                args.remove(0);
            }
            s if s.starts_with('-') => {
                eprintln!("unknown option: {s}");
                print_usage();
                process::exit(2);
            }
            _ => {
                paths.push(args.remove(0));
            }
        }
    }

    if use_stdin && !paths.is_empty() {
        eprintln!("--stdin cannot be combined with file paths");
        process::exit(2);
    }
    if !use_stdin && paths.is_empty() {
        print_usage();
        process::exit(2);
    }

    let mut any_blocked = false;
    let mut hard_error = false;

    if use_stdin {
        let mut buf = String::new();
        if let Err(e) = io::stdin().read_to_string(&mut buf) {
            eprintln!("stdin read error: {e}");
            process::exit(2);
        }
        match scan_one("<stdin>", &buf, json, verbose) {
            Ok(true) => {}
            Ok(false) => any_blocked = true,
            Err(msg) => {
                eprintln!("{msg}");
                hard_error = true;
            }
        }
    } else {
        for p in &paths {
            let path = Path::new(p);
            if !path.is_file() {
                eprintln!("[ERROR] not a file: {p}");
                hard_error = true;
                continue;
            }
            match fs::read_to_string(path) {
                Ok(content) => match scan_one(p, &content, json, verbose) {
                    Ok(true) => {}
                    Ok(false) => any_blocked = true,
                    Err(msg) => {
                        eprintln!("{msg}");
                        hard_error = true;
                    }
                },
                Err(e) => {
                    eprintln!("[ERROR] cannot read {p}: {e}");
                    hard_error = true;
                }
            }
        }
    }

    if hard_error {
        process::exit(2);
    }
    if any_blocked {
        process::exit(1);
    }
    process::exit(0);
}
