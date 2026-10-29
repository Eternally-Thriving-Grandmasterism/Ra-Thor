//! Audit files with FENCA and print the reports as JSON.
//!
//! `audit_score` is the fraction of the four mercy checks that passed.
//! This binary does not print a valence constant.
//!
//! Exit 0 when every file passed.
//! Exit 1 when at least one audit did not pass (JSON is still printed).
//! Exit 2 when an input is missing, unreadable, or not valid UTF-8.

use std::env;
use std::fs;
use std::pin::Pin;
use std::process::ExitCode;
use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

use fenca::FencaEternalCheck;

fn block_on<F: std::future::Future>(future: F) -> F::Output {
    fn noop(_: *const ()) {}
    fn clone(ptr: *const ()) -> RawWaker {
        RawWaker::new(ptr, &VTABLE)
    }
    static VTABLE: RawWakerVTable = RawWakerVTable::new(clone, noop, noop, noop);
    let waker = unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &VTABLE)) };
    let mut cx = Context::from_waker(&waker);
    let mut future = Pin::from(Box::pin(future));
    match future.as_mut().poll(&mut cx) {
        Poll::Ready(output) => output,
        Poll::Pending => panic!("fenca audit future must be ready without waiting"),
    }
}

fn json_string(value: &str) -> String {
    let mut out = String::from("\"");
    for c in value.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Print a quarter fraction exactly. Any other value is printed raw so a
/// caller can reject it. This function never substitutes a passing constant.
fn json_score(score: f64) -> String {
    const QUARTERS: [(f64, &str); 5] = [
        (0.0, "0.0"),
        (0.25, "0.25"),
        (0.5, "0.5"),
        (0.75, "0.75"),
        (1.0, "1.0"),
    ];
    for (value, label) in QUARTERS {
        if (score - value).abs() < 1e-9 {
            return label.to_string();
        }
    }
    format!("{score}")
}

fn main() -> ExitCode {
    let paths: Vec<String> = env::args().skip(1).collect();
    if paths.is_empty() {
        eprintln!("missing input: no files to audit");
        return ExitCode::from(2);
    }

    let mut body = String::from("[");
    let mut any_failed = false;
    for (index, path) in paths.iter().enumerate() {
        let text = match fs::read_to_string(path) {
            Ok(text) => text,
            Err(err) => {
                eprintln!("missing or unreadable input: {path}: {err}");
                return ExitCode::from(2);
            }
        };
        let report = block_on(FencaEternalCheck::verify_codex_content(&text));
        if !report.passed {
            any_failed = true;
        }
        if index > 0 {
            body.push(',');
        }
        body.push_str("{\"path\":");
        body.push_str(&json_string(path));
        body.push_str(",\"passed\":");
        body.push_str(if report.passed { "true" } else { "false" });
        body.push_str(",\"audit_score\":");
        body.push_str(&json_score(report.audit_score));
        body.push_str(",\"checks_performed\":[");
        for (check_index, check) in report.checks_performed.iter().enumerate() {
            if check_index > 0 {
                body.push(',');
            }
            body.push_str(&json_string(check));
        }
        body.push_str("],\"reasons\":[");
        for (reason_index, reason) in report.reasons.iter().enumerate() {
            if reason_index > 0 {
                body.push(',');
            }
            body.push_str(&json_string(reason));
        }
        body.push_str("]}");
    }
    body.push(']');
    println!("{body}");
    if any_failed {
        ExitCode::from(1)
    } else {
        ExitCode::from(0)
    }
}
