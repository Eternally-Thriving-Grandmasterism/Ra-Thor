# Lived-tick reader (T1)

**Crate:** `reality-thriving-transfer`  
**Door:** T1 (PATSAGi R7)  
**Contact:** info@Rathor.ai  
**Workspace:** 14.15.6

## What this is

Optional filesystem reader for `powrush_lived_tick_v1` JSON produced by the Powrush yard when ingest is on **there**. This lattice seat only **reads** for R&D mercy summaries.

## What this is not

- Not the RTT / bridging contract (`POWRUSH_TELEMETRY_CONTRACT.md`)
- Not a flip of `POWRUSH_INGEST`
- Not a key driver (no WASD / Title Online / Market)
- Not a new workspace member

## Env

| Variable | Behavior |
|----------|----------|
| `POWRUSH_LIVED_TICK_PATH` unset or empty | No-op (`Ok(None)`) |
| set to a file path | Load JSON → one-line mercy summary |

## API

- `parse_powrush_lived_tick_json`
- `mercy_summary_line`
- `load_lived_tick_from_path`
- `load_lived_tick_from_env`

```bash
cargo test -p reality-thriving-transfer lived_tick
```
