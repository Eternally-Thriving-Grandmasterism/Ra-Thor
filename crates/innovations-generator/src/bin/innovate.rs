//! `innovate` reads base ideas as JSON and writes at most three ranked seeds.
//! `innovate screen-name` runs deterministic character-name Mercy-Checks.

use std::env;
use std::fs;
use std::io::{self, Read, Write};
use std::process::ExitCode;

use innovations_generator::{
    innovate, parse_base_ideas, render_name_report, render_report, screen_names,
};

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(err) => {
            eprintln!("innovate: {err}");
            ExitCode::from(1)
        }
    }
}

fn run() -> Result<ExitCode, String> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        print_help();
        return Ok(ExitCode::SUCCESS);
    }
    if args.first().map(String::as_str) == Some("screen-name") {
        return run_screen_name(&args[1..]);
    }
    if args.len() > 1 {
        return Err("pass one file path, or no arguments to read stdin (see innovate screen-name)"
            .into());
    }
    let raw = read_input(args.first().map(String::as_str))?;
    let ideas = parse_base_ideas(&raw)?;
    let seeds = innovate(&ideas);
    let json = render_report(&seeds)?;
    let mut stdout = io::stdout().lock();
    stdout
        .write_all(json.as_bytes())
        .map_err(|err| format!("failed to write stdout: {err}"))?;
    Ok(ExitCode::SUCCESS)
}

fn run_screen_name(args: &[String]) -> Result<ExitCode, String> {
    let mut cast: Vec<String> = Vec::new();
    let mut names: Vec<String> = Vec::new();
    let mut index = 0;
    while index < args.len() {
        let arg = &args[index];
        if arg == "--cast" {
            index += 1;
            let value = args
                .get(index)
                .ok_or_else(|| "--cast requires a comma-separated name list".to_string())?;
            cast = value
                .split(',')
                .map(str::trim)
                .filter(|part| !part.is_empty())
                .map(String::from)
                .collect();
            index += 1;
            continue;
        }
        if arg.starts_with("--cast=") {
            let value = arg.trim_start_matches("--cast=");
            cast = value
                .split(',')
                .map(str::trim)
                .filter(|part| !part.is_empty())
                .map(String::from)
                .collect();
            index += 1;
            continue;
        }
        names.push(arg.clone());
        index += 1;
    }
    if names.is_empty() {
        return Err("screen-name: pass at least one candidate name".into());
    }
    let (results, any_fail) = screen_names(&names, &cast);
    let mut stdout = io::stdout().lock();
    for result in &results {
        let block = render_name_report(result);
        stdout
            .write_all(block.as_bytes())
            .map_err(|err| format!("failed to write stdout: {err}"))?;
    }
    Ok(if any_fail {
        ExitCode::from(1)
    } else {
        ExitCode::SUCCESS
    })
}

fn read_input(path: Option<&str>) -> Result<String, String> {
    match path {
        None | Some("-") => {
            let mut buf = String::new();
            io::stdin()
                .read_to_string(&mut buf)
                .map_err(|err| format!("failed to read stdin: {err}"))?;
            Ok(buf)
        }
        Some(path) => {
            fs::read_to_string(path).map_err(|err| format!("failed to read {path}: {err}"))
        }
    }
}

fn print_help() {
    println!(
        "\
innovate — rank at most 3 idea seeds from JSON base ideas

Usage:
  innovate [FILE]
  innovate -              read stdin
  innovate screen-name [--cast name1,name2,...] <NAME>...
  innovate --help

Seed input is a JSON array of strings or objects, or {{\"base_ideas\": [...]}}.
Each object may set \"text\" (or \"summary\" or \"title\"), \"sources\", and \"evidence\".
Output JSON has seeds with rank, title, summary, claim_tier, sources, and kill_test.
claim_tier is one of: idea, plausible, evidenced.

screen-name runs deterministic character-name Mercy-Checks (English heuristic lists).
It prints PASS / FAIL / FLAG lines per name and always prints NEEDS reminder lines.
Exit code 1 when any hard check FAILs or on usage / I/O errors.
"
    );
}
