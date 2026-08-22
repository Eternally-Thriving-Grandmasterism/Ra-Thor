//! agsi-eval-rg — Thin runnable subject for AGSi-eval Slice B.
//!
//! Default subject is R (lattice gates). G and RG print a NOT_BOUND report.
//!
//!   cargo run -p mercy-security --bin agsi-eval-rg -- --items science/agsi-eval/slice_b/items.json
//!
//! Contact: info@Rathor.ai | AG-SML v1.0

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;

use mercy_security::agsi_eval::{evaluate_slice_r, unbound_report, EvalSubject, SliceItem};

fn usage() {
    eprintln!(
        "Usage: agsi-eval-rg [--subject R|G|RG] [--repo-root PATH] --items PATH\n\
         \n\
         Subject R  runs lattice gates (IngestionScanner + HarmRefusal + harness).\n\
         Subject G  and RG are NOT_BOUND (no model adapter in this crate).\n\
         Emits one JSON SliceBReport on stdout.\n\
         Contact: info@Rathor.ai"
    );
}

fn parse_subject(s: &str) -> Option<EvalSubject> {
    match s.to_ascii_uppercase().as_str() {
        "R" => Some(EvalSubject::R),
        "G" => Some(EvalSubject::G),
        "RG" => Some(EvalSubject::Rg),
        _ => None,
    }
}

fn main() {
    let mut args: Vec<String> = env::args().skip(1).collect();
    let mut subject = EvalSubject::R;
    let mut repo_root = PathBuf::from(".");
    let mut items_path: Option<PathBuf> = None;

    while let Some(a) = args.first().cloned() {
        match a.as_str() {
            "-h" | "--help" => {
                usage();
                process::exit(0);
            }
            "--subject" => {
                args.remove(0);
                let v = args.first().cloned().unwrap_or_default();
                args.remove(0);
                match parse_subject(&v) {
                    Some(s) => subject = s,
                    None => {
                        eprintln!("unknown subject: {v}");
                        process::exit(2);
                    }
                }
            }
            "--repo-root" => {
                args.remove(0);
                let v = args.first().cloned().unwrap_or_default();
                args.remove(0);
                repo_root = PathBuf::from(v);
            }
            "--items" => {
                args.remove(0);
                let v = args.first().cloned().unwrap_or_default();
                args.remove(0);
                items_path = Some(PathBuf::from(v));
            }
            s if s.starts_with('-') => {
                eprintln!("unknown option: {s}");
                usage();
                process::exit(2);
            }
            _ => {
                eprintln!("unexpected argument: {a}");
                usage();
                process::exit(2);
            }
        }
    }

    if !subject.is_bound() {
        let report = unbound_report(subject);
        println!("{}", serde_json::to_string_pretty(&report).unwrap());
        process::exit(0);
    }

    let items_path = match items_path {
        Some(p) => {
            if p.is_absolute() {
                p
            } else {
                repo_root.join(p)
            }
        }
        None => {
            usage();
            process::exit(2);
        }
    };

    let raw = match fs::read_to_string(&items_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("cannot read items {}: {e}", items_path.display());
            process::exit(2);
        }
    };
    let items: Vec<SliceItem> = match serde_json::from_str(&raw) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("items JSON error: {e}");
            process::exit(2);
        }
    };

    let root = repo_root.clone();
    let report = evaluate_slice_r(&items, |rel| {
        let p = if Path::new(rel).is_absolute() {
            PathBuf::from(rel)
        } else {
            root.join(rel)
        };
        fs::read_to_string(&p).map_err(|e| format!("{e}"))
    });

    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    if report.leaks > 0 {
        process::exit(1);
    }
    process::exit(0);
}
