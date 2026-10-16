//! `innovate` reads base ideas as JSON and writes at most three ranked seeds.

use std::env;
use std::fs;
use std::io::{self, Read, Write};
use std::process::ExitCode;

use innovations_generator::{innovate, parse_base_ideas, render_report};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("innovate: {err}");
            ExitCode::from(1)
        }
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!(
            "\
innovate — rank at most 3 idea seeds from JSON base ideas

Usage:
  innovate [FILE]
  innovate -              read stdin
  innovate --help

Input is a JSON array of strings or objects, or {{\"base_ideas\": [...]}}.
Each object may set \"text\" (or \"summary\" or \"title\"), \"sources\", and \"evidence\".
Output JSON has seeds with rank, title, summary, claim_tier, sources, and kill_test.
claim_tier is one of: idea, plausible, evidenced.
"
        );
        return Ok(());
    }
    if args.len() > 1 {
        return Err("pass one file path, or no arguments to read stdin".into());
    }
    let raw = match args.first().map(String::as_str) {
        None | Some("-") => {
            let mut buf = String::new();
            io::stdin()
                .read_to_string(&mut buf)
                .map_err(|err| format!("failed to read stdin: {err}"))?;
            buf
        }
        Some(path) => {
            fs::read_to_string(path).map_err(|err| format!("failed to read {path}: {err}"))?
        }
    };
    let ideas = parse_base_ideas(&raw)?;
    let seeds = innovate(&ideas);
    let json = render_report(&seeds)?;
    let mut stdout = io::stdout().lock();
    stdout
        .write_all(json.as_bytes())
        .map_err(|err| format!("failed to write stdout: {err}"))?;
    Ok(())
}
