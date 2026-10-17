//! CLI coverage for `innovate screen-name`.

use std::process::Command;

fn innovate_bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_innovate"))
}

#[test]
fn screen_name_coil_fails_with_exit_1() {
    let output = innovate_bin()
        .args(["screen-name", "Coil"])
        .output()
        .expect("run screen-name");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Character-ness: FAIL"));
    assert!(stdout.contains(NEEDS_RESEARCH_LINE));
    assert_eq!(output.status.code(), Some(1));
}

#[test]
fn screen_name_hesper_succeeds() {
    let output = innovate_bin()
        .args(["screen-name", "Hesper"])
        .output()
        .expect("run screen-name");
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Sound-alikes: PASS"));
}

#[test]
fn screen_name_empty_whitespace_fails_with_exit_1() {
    let output = innovate_bin()
        .args(["screen-name", "   "])
        .output()
        .expect("run screen-name");
    assert_eq!(output.status.code(), Some(1));
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("empty after trim"));
}

const NEEDS_RESEARCH_LINE: &str =
    "NEEDS RESEARCH: IP/trademark and own-repo clash search";
