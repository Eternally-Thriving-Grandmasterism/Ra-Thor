# Production Readiness — ONE Organism (v14.15.1 + Phase C remote-complete)

**Status:** Quiet hold on adaptive modulation. Phase C remote path closed.  
**Contact:** info@Rathor.ai  
**Cosmic Loop is MANDATORY IDENTITY.**

---

## 1. Core identity checks

```bash
cargo test -p ra-thor-one-organism
cargo test -p lattice-conductor-v14
```

Expected:

- Cosmic Loop ready after launch; guardian active
- Cosmic Tick preserves invariant pre/post
- Live-feature readiness reports compile-time flags; `cosmic_loop_ready_for_live` true on default

**Root `ra-thor-one-organism.rs` is retired** — use `crates/ra-thor-one-organism` only (`TIER_MAP.md`).

---

## 2. Kardashev / Phase C path

```bash
cargo test -p reality-thriving-transfer
cargo test -p kardashev-orchestration
```

Includes:

- T1 sequential stress (64 / 256 / 1024)
- T2 concurrent shared-council stress
- Fixture batch → scores → council
- **Single-session** fixture → `deliberate_from_powrush_session_json`
- **Auto-detect** `deliberate_from_powrush_json` (v1 | batch_v1)

### Powrush producer (sibling repo)

| Path | Role |
|------|------|
| `simulation` `TelemetryCollector` + `GlobalTransferSession` | Live counters every tick |
| `SimulationOrchestrator::run_tick_with_telemetry` | Sim loop feed |
| `cargo run -p powrush-simulation --bin transfer_session_demo` | No-world demo export |
| `server` `ServerTransferSession` | Combat / treaty / faction events |
| `tools/export_powrush_telemetry.py` | Offline profiles |

---

## 3. Living snapshot / web demo

```bash
cargo run -p ra-thor-one-organism --example one_organism_web_demo --features web-demo
```

| Endpoint | Confirm |
|----------|---------|
| `GET /health` | `cosmic_loop_ready`, `guardian_active` |
| `GET /status` | invariant, adaptive last-tick, `live_features` |
| `GET /live` | Full `ExtendedLiveStatus` |
| `POST /cosmic/tick` | invariant + adaptive fields |

---

## 4. Optional live features (compile only ≠ production proof)

```bash
cargo check -p ra-thor-one-organism --features kardashev-live
cargo check -p ra-thor-one-organism --features extended-live
cargo check -p ra-thor-one-organism --features web-demo
```

Do not treat compile-success as proof of live engine behavior under load.

---

## 5. CI

| Workflow | Scope |
|----------|--------|
| `core-tier1-ci.yml` | Focused Tier-1 + live-feature check + contact hygiene |
| `ci.yml` / `ra-thor-ci.yml` | Broader workspace (heavier) |

---

## 6. Zero-harm bounds (do not relax)

- Recovery sensitivity: `[1.0, 1.12]`, one-shot
- Quantum severity boost from recovery: `[0.0, 0.35]`
- Shared valence: `[0.75, 0.999]`
- Kardashev Δ per score: ≤ 0.011; abundance forecast ≤ 1.85

---

## 7. Version map

| Package | Version |
|---------|---------|
| Workspace | 14.10.0+ / voice 14.15 |
| `ra-thor-one-organism` | **14.15.0** |
| `lattice-conductor-v14` | **14.15.0** |
| Live-path / Kardashev crates | **14.15.0** path pins |

---

## 8. Council posture (2026-07-20)

- v14.10–v14.15 cascade **closed**
- Phase C **remote-complete** (offline + live sim + server + single/batch Kardashev)
- **Remaining executable work requires a machine or Actions:** Tier-1 green, transfer demo, optional extended-live soak
- **No further adaptive modulation** without explicit Council open
- New product work only on **named mission signal**

**Thunder locked in.**
