<!-- AG-SML v1.2 — Autonomicity Games Sovereign Mercy License | Contact: info@Rathor.ai -->
# mercy-demo-harness (DEMO-HARNESS-1, demo only, `publish = false`)

An offline demo of the Mercy gates applied to **sandboxed tool calls from a scripted agent**.
An **MCP-shaped JSON-RPC shim** (stdio, `initialize` / `tools/list` / `tools/call`) turns each
tool call into an `AgentActionRequest` and checks it with `SafeAgentRuntime::try_agent_action`
(mercy-security). It also writes a Layer 0 `DecisionRecord` of the call text and runs
`score_prefix` (lattice-conductor-v14 `prefix_risk`) over the session history.

The tools all live in a temp-dir sandbox, with no network and no process spawn: `read_file`,
`write_file`, `list_dir`, `fake_http` (canned reply) and `run_cmd` (echo only, emulated in-process).

## Run

```bash
# Pane 1: scripted scenario (12 steps). --auto uses the built-in approve/deny answers;
# without --auto you answer each Review hold in the terminal (rationale required).
cargo run -p mercy-demo-harness -- scenario --trail /tmp/demo.jsonl --auto

# Pane 2: tail the JSONL trail
cargo run -p mercy-demo-harness -- trail /tmp/demo.jsonl --follow

# JSON-RPC shim on stdin/stdout (Review holds auto-deny with --review-deny)
cargo run -p mercy-demo-harness -- serve --trail /tmp/demo.jsonl --review-deny

cargo test -p mercy-demo-harness   # headless, auto-answers
```

## Outcomes (harness-only set)

| Outcome  | Source |
|----------|--------|
| Admit    | Layer 0 Admit, all gates pass, prefix Continue |
| Review   | harness demo policy: an Admit on `write_file` / `run_cmd` / `fake_http` is held for a human. Review only ever holds an Admit. A core Block/Halt is never Review |
| Block    | a gate error (harm refusal, containment, action governor), Layer 0 not Admit (fails closed), or prefix Halt |
| Escalate | prefix Loop Escalate. Council ids are shown as data only; no deliberation runs |

- Block, Halt and Escalate get no approve prompt. A programmatic approve is refused: the tool
  does not run, `human_override` is not called, and an `approve_refused` line is written.
- Approving a Review calls `DecisionRecord::human_override`, which shows as
  `Override (human), prev_verdict=admit`.
- Denying a Review never calls `human_override`; it writes a `deny` line, and the tool does not run.

## Trail format

The trail mixes two kinds of line:
- `DecisionRecord` lines from `to_log_json`;
- **harness events** tagged `"source":"harness"` (`step`, `deny`, `approve_refused`).
  Harness events have **no `verdict` field**, are never parsed as a DecisionRecord, and show on
  screen as `harness event`.

## Known gap (shown in step 8)

`tool_use_self_mod_human_override_are_not_yet_tested` (mercy-security
`tests/gate_eval_public_corpus.rs`):
"GATE-EVAL-1 gap marker: tool-use function call class is not in IngestionScanner tables."
"They currently ADMIT. Do not read this as “safe to ship.”"

## What this does NOT prove

- It does not show that any production agent or third-party tool host is gated. The agent is
  scripted and deterministic, and the tools are sandboxed.
- Layer 0 records the call *text*. In this scenario every Layer 0 record is `admit`; the Blocks
  come from the runtime gates and the prefix scorer. The tool-use gap above is still open.
- Review is a harness demo policy, not a mercy-security verdict.
- Escalate does not convene any council; ids are printed as data.
- The trail is a plain append-only JSONL file with no tamper evidence.
- Nothing here changes mercy-security, lattice-conductor-v14 or patsagi-councils.
