//! mercy-demo-harness CLI (demo only). AG-SML v1.2 | Contact: info@Rathor.ai
//!
//!   mercy-demo-harness scenario [--trail PATH] [--auto]   scripted agent, ~12 steps
//!   mercy-demo-harness serve --trail PATH [--review-deny]  JSON-RPC lines on stdin/stdout
//!   mercy-demo-harness trail PATH [--follow]               render the JSONL trail

use std::io::{BufRead, Read, Seek, SeekFrom};
use std::time::Duration;

use mercy_demo_harness::*;

fn arg_after(args: &[String], flag: &str) -> Option<String> {
    args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1).cloned())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(String::as_str).unwrap_or("help");
    let trail = arg_after(&args, "--trail").unwrap_or_else(|| "mercy-demo-trail.jsonl".into());
    match cmd {
        "scenario" => {
            let mut h = Harness::new(&trail).expect("sandbox");
            let mut auto = scenario_auto_answers();
            let mut tty = TerminalAnswers;
            let answers: &mut dyn Answerer = if args.iter().any(|a| a == "--auto") { &mut auto } else { &mut tty };
            println!("mercy-demo-harness: sandboxed tool calls from a scripted agent (demo only)");
            println!("trail: {trail}   profile: {DEMO_PROFILE_NAME}   sandbox: {}\n", h.sandbox.root.display());
            for (i, s) in scenario().iter().enumerate() {
                let (_resp, o) = h.handle_rpc(&rpc_line(i + 1, s), answers);
                if let Some(o) = o { println!("{}\n", render_step(s, &o)); }
            }
        }
        "serve" => {
            let mut h = Harness::new(&trail).expect("sandbox");
            let mut deny = AutoAnswers(vec![]);
            let mut tty = TerminalAnswers;
            let answers: &mut dyn Answerer = if args.iter().any(|a| a == "--review-deny") { &mut deny } else { &mut tty };
            for line in std::io::stdin().lock().lines().map_while(Result::ok) {
                if line.trim().is_empty() { continue; }
                let (resp, _) = h.handle_rpc(&line, answers);
                println!("{resp}");
            }
        }
        "trail" => {
            let path = args.get(2).cloned().unwrap_or(trail);
            let follow = args.iter().any(|a| a == "--follow");
            let mut pos = 0u64;
            let mut buf = String::new();
            loop {
                if let Ok(mut f) = std::fs::File::open(&path) {
                    let _ = f.seek(SeekFrom::Start(pos));
                    let mut chunk = String::new();
                    let _ = f.read_to_string(&mut chunk);
                    pos += chunk.len() as u64;
                    buf.push_str(&chunk);
                    while let Some(i) = buf.find('\n') {
                        let line: String = buf.drain(..=i).collect();
                        if !line.trim().is_empty() { println!("{}", render_trail_line(line.trim())); }
                    }
                }
                if !follow { break; }
                std::thread::sleep(Duration::from_millis(250));
            }
        }
        _ => {
            eprintln!("usage: mercy-demo-harness scenario [--trail PATH] [--auto] | serve --trail PATH [--review-deny] | trail PATH [--follow]");
        }
    }
}
