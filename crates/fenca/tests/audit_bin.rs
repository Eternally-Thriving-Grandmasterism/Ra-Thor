//! The audit binary reports a real fraction and can exit non-zero.

use std::fs;
use std::process::Command;

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_fenca_audit")
}

fn write_temp(name: &str, contents: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!("fenca-audit-{name}-{}", std::process::id()));
    fs::write(&path, contents).expect("write temp");
    path
}

#[test]
fn known_good_file_exits_zero_with_score_one() {
    let path = write_temp(
        "good",
        "Research note. A human reviews this draft before anyone relies on it.\n",
    );
    let output = Command::new(bin()).arg(&path).output().expect("run bin");
    let stdout = String::from_utf8(output.stdout).expect("utf-8");
    assert_eq!(
        output.status.code(),
        Some(0),
        "stderr {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(stdout.contains("\"passed\":true"), "{stdout}");
    assert!(stdout.contains("\"audit_score\":1.0"), "{stdout}");
    assert!(stdout.contains("\"reasons\":[]"), "{stdout}");
    assert!(!stdout.contains("0.999999"), "{stdout}");
    let _ = fs::remove_file(path);
}

#[test]
fn banned_marker_exits_one_with_three_quarter_score() {
    let path = write_temp("bad", "draft claims perfect valence\n");
    let output = Command::new(bin()).arg(&path).output().expect("run bin");
    let stdout = String::from_utf8(output.stdout).expect("utf-8");
    assert_eq!(output.status.code(), Some(1), "{stdout}");
    assert!(stdout.contains("\"passed\":false"), "{stdout}");
    assert!(stdout.contains("\"audit_score\":0.75"), "{stdout}");
    assert!(stdout.contains("perfect valence"), "{stdout}");
    assert!(!stdout.contains("0.999999"), "{stdout}");
    let _ = fs::remove_file(path);
}

#[test]
fn missing_input_exits_two_and_prints_no_score() {
    let missing = std::env::temp_dir().join(format!("fenca-audit-missing-{}", std::process::id()));
    let _ = fs::remove_file(&missing);
    let output = Command::new(bin()).arg(&missing).output().expect("run bin");
    assert_eq!(output.status.code(), Some(2));
    assert!(
        output.stdout.is_empty(),
        "stdout {}",
        String::from_utf8_lossy(&output.stdout)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("missing or unreadable input"), "{stderr}");

    let no_args = Command::new(bin()).output().expect("run bin");
    assert_eq!(no_args.status.code(), Some(2));
    assert!(no_args.stdout.is_empty());
}
