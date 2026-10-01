//! TIER1-AGSI-2 — agsi-eval-rg argument handling: a flag with no value must
//! print usage and exit 2 (never panic with exit 101).
//! Contact: info@Rathor.ai

use std::process::Command;

const VALUE_FLAGS: &[&str] = &[
    "--items",
    "--slice",
    "--subject",
    "--adapter",
    "--model-id",
    "--log",
    "--repo-root",
];

#[test]
fn flag_without_value_prints_usage_and_exits_2() {
    let bin = env!("CARGO_BIN_EXE_agsi-eval-rg");
    for flag in VALUE_FLAGS {
        let out = Command::new(bin).arg(flag).output().expect("run agsi-eval-rg");
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert_eq!(out.status.code(), Some(2), "{flag}: stderr={stderr}");
        assert!(stderr.contains("Usage: agsi-eval-rg"), "{flag}: stderr={stderr}");
        assert!(!stderr.contains("panicked"), "{flag}: stderr={stderr}");
    }
}

#[test]
fn documented_items_path_exists() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    for rel in [
        "docs/archive/root-dirs/research/science/agsi-eval/slice_b/items.json",
        "docs/archive/root-dirs/research/science/agsi-eval/slice_b1/items.json",
        "docs/archive/root-dirs/research/science/agsi-eval/slice_b/wrap_items.json",
    ] {
        assert!(root.join(rel).is_file(), "missing {rel}");
        let src = include_str!("../src/bin/agsi_eval_rg.rs");
        assert!(src.contains(rel), "usage docs should point at {rel}");
    }
}
