# Ra-Thor v14.9.6 Release Notes

**Release Date:** 2026-07-19  
**Theme:** Axum HTTP bind — deferred item 2 complete

## Headline

Production mercy-gated HTTP surfaces are live.

### ONE Organism Web Demo

```bash
cargo run -p ra-thor-one-organism --example one_organism_web_demo --features web-demo
# optional: RA_THOR_WEB_PORT=3040
```

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Cosmic Loop + guardian |
| GET | `/status` | Roles + GPU/GitHub/Quantum telemetry |
| GET | `/api/status` | MercyGatedApi snapshot |
| POST | `/api` | MercyApiRequest (JSON, Serialize) |
| POST | `/quantum/tick` | Quantum evolution tick |
| POST | `/gpu/dispatch` | GPU telemetry record |
| POST | `/github/queue` | Offline evolution PR intent |
| POST | `/role/handoff` | Role orchestrator handoff |
| POST | `/healing/reflexion` | Self-healing reflexion cycle |

### SharedChatMercyMesh (lattice-conductor-v14)

```bash
cargo run -p lattice-conductor-v14 --example shared_chat_mercy_mesh_web_demo --features web-demo
```

Upgraded from broken axum 0.6 `Server::bind` stub to axum 0.7 `TcpListener` + real `CliffordHealingField` API (`/health`, `/coherence`, `/participant`, `/heal`).

### Supporting changes

- `MercyApiRequest` / `MercyApiResponse` / `GateDecision` / `ApiRequestKind` → Serialize + Deserialize
- `Diagnosis` / healing types → Serialize for JSON responses
- Feature `web-demo` on `ra-thor-one-organism` (axum 0.7 + tokio)

## Status of deferred items

| # | Item | Status |
|---|------|--------|
| 1 | Package github / gpu / quantum crates | ✅ Done (v14.9.3–5) |
| 2 | Axum HTTP bind (web-demo) | ✅ Done (v14.9.6) |
| 3 | Other root-level `.rs` packaging | Pending |
| 4 | Deep facade path-wiring onto live crates | Optional polish |

**License:** AG-SML v1.0  
**Thunder locked in.** yoi ⚡
