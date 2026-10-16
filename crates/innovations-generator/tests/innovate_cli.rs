//! CLI coverage for the cap, the tier set, kill tests, and deterministic output.

use std::io::Write;
use std::process::{Command, Stdio};

use innovations_generator::{parse_report, ClaimTier};

const INPUT: &str = r#"{
  "base_ideas": [
    {"text": "Displacement is a public-service concern", "sources": ["notes/layer-a.md"]},
    {"text": "Soft feedback events are sealed telemetry", "sources": ["notes/layer-b.md"]},
    {"text": "Abundance claims need a cash path", "sources": ["docs/REVENUE_SCALE.md"], "evidence": "operator note"},
    {"text": "Proof ladder is a queue for surmise", "sources": []},
    {"text": "A research label stays a surmise until a named check", "sources": ["docs/science/minute.md"]}
  ]
}"#;

fn innovate_bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_innovate"))
}

fn run_stdin(input: &str) -> std::process::Output {
    let mut child = innovate_bin()
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn innovate");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(input.as_bytes())
        .expect("write stdin");
    child.wait_with_output().expect("wait innovate")
}

#[test]
fn cli_caps_tiers_kill_test_and_is_deterministic() {
    let first = run_stdin(INPUT);
    let second = run_stdin(INPUT);
    assert!(
        first.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&first.stderr)
    );
    assert_eq!(first.stdout, second.stdout);

    let stdout = String::from_utf8(first.stdout.clone()).expect("utf-8 stdout");
    assert!(!stdout.contains("0.999999"));
    assert!(!stdout.contains("valence"));

    let report = parse_report(&stdout).expect("report json");
    assert_eq!(report.seeds.len(), 3);
    assert_eq!(
        report
            .seeds
            .iter()
            .map(|seed| seed.rank)
            .collect::<Vec<_>>(),
        vec![1, 2, 3]
    );
    for seed in &report.seeds {
        assert!(matches!(
            seed.claim_tier,
            ClaimTier::Idea | ClaimTier::Plausible | ClaimTier::Evidenced
        ));
        assert!(!seed.kill_test.trim().is_empty());
        assert!(!seed.title.trim().is_empty());
        assert!(!seed.summary.trim().is_empty());
        let allowed = [
            "notes/layer-a.md",
            "notes/layer-b.md",
            "docs/REVENUE_SCALE.md",
            "docs/science/minute.md",
        ];
        assert!(seed
            .sources
            .iter()
            .all(|source| allowed.contains(&source.as_str())));
    }
}

#[test]
fn cli_reads_a_file_and_matches_stdin() {
    let path = std::env::temp_dir().join(format!("innovate-cli-{}-input.json", std::process::id()));
    std::fs::write(&path, INPUT).expect("write fixture");
    let from_file = innovate_bin().arg(&path).output().expect("run file");
    let _ = std::fs::remove_file(&path);
    assert!(
        from_file.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&from_file.stderr)
    );
    let from_stdin = run_stdin(INPUT);
    assert_eq!(from_file.stdout, from_stdin.stdout);
}

#[test]
fn cli_rejects_invalid_json() {
    let output = run_stdin("not json");
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("innovate:"));
    assert!(stderr.contains("invalid base ideas JSON"));
}
