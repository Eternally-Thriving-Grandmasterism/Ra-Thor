//! agsi-eval-rg — Runnable subjects for AGSi-eval Slice B.
//!
//!   --subject R --items science/agsi-eval/slice_b/items.json
//!   --subject RG --adapter item --items science/agsi-eval/slice_b/wrap_items.json
//!   --subject RG --adapter file:science/agsi-eval/slice_b/CANDIDATES.example.json --items science/agsi-eval/slice_b/wrap_items.json
//!   --subject RG --adapter echo   # smoke only, circular with R
//!   --subject G                   # NOT_BOUND
//!
//! Contact: info@Rathor.ai | AG-SML v1.0

use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;

use mercy_security::agsi_eval::{
    evaluate_slice_r, evaluate_slice_rg, unbound_report, EchoAdapter, EvalSubject, FileAdapter,
    ItemCandidateAdapter, SliceItem,
};

fn usage() {
    eprintln!(
        "Usage: agsi-eval-rg [--subject R|G|RG] [--adapter none|echo|item|file:PATH] [--repo-root PATH] --items PATH\n\
         \n\
         R                 lattice gates on item prompts/fixtures (B.0)\n\
         RG --adapter item wrap using each item's `candidate` field (distinct)\n\
         RG --adapter file:PATH  wrap using id→candidate map\n\
         RG --adapter echo smoke only (circular with R)\n\
         G                 NOT_BOUND\n\
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
    let mut adapter_spec = "none".to_string();

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
            "--adapter" => {
                args.remove(0);
                adapter_spec = args.first().cloned().unwrap_or_default();
                args.remove(0);
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

    if matches!(subject, EvalSubject::G) {
        println!("{}", serde_json::to_string_pretty(&unbound_report(subject)).unwrap());
        process::exit(0);
    }

    if matches!(subject, EvalSubject::Rg) && adapter_spec == "none" {
        println!("{}", serde_json::to_string_pretty(&unbound_report(subject)).unwrap());
        process::exit(0);
    }

    let items_path = match items_path {
        Some(p) if p.is_absolute() => p,
        Some(p) => repo_root.join(p),
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
    let load = |rel: &str| {
        let p = if Path::new(rel).is_absolute() {
            PathBuf::from(rel)
        } else {
            root.join(rel)
        };
        fs::read_to_string(&p).map_err(|e| format!("{e}"))
    };

    let report = match subject {
        EvalSubject::R => evaluate_slice_r(&items, load),
        EvalSubject::Rg if adapter_spec == "echo" => evaluate_slice_rg(&items, &EchoAdapter, load),
        EvalSubject::Rg if adapter_spec == "item" => {
            evaluate_slice_rg(&items, &ItemCandidateAdapter, load)
        }
        EvalSubject::Rg if adapter_spec.starts_with("file:") => {
            let path = adapter_spec.trim_start_matches("file:");
            let p = if Path::new(path).is_absolute() {
                PathBuf::from(path)
            } else {
                repo_root.join(path)
            };
            let txt = match fs::read_to_string(&p) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("cannot read adapter file {}: {e}", p.display());
                    process::exit(2);
                }
            };
            let map: HashMap<String, String> = match serde_json::from_str(&txt) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("adapter file JSON must be {{id: candidate}}: {e}");
                    process::exit(2);
                }
            };
            evaluate_slice_rg(&items, &FileAdapter { map }, load)
        }
        EvalSubject::Rg => {
            eprintln!("RG requires --adapter echo | item | file:PATH");
            process::exit(2);
        }
        EvalSubject::G => unreachable!(),
    };

    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    if report.leaks > 0 {
        process::exit(1);
    }
    process::exit(0);
}
