# Lattice Conductor v13 — **DEPRECATED**

> **Do not start new work on this crate.**
>
> Canonical lineage is **`lattice-conductor-v14`**.
> For `Conductable` / `MercyAligned` / `SimpleLatticeConductor`, enable:
>
> ```toml
> lattice-conductor-v14 = { path = "../lattice-conductor-v14", features = ["v13-compat"] }
> ```
>
> Full strategy: [`crates/lattice-conductor-v14/MIGRATION_v13_to_v14.md`](../lattice-conductor-v14/MIGRATION_v13_to_v14.md)

This crate remains in the workspace for **one or two more cycles** so existing paths do not break. It will move to `archive/` or a legacy group after Phase 2 consumers are fully green.

Contact: info@Rathor.ai

---

## Historical notes (v13.x)

### v13.1 Foundation
- `metta_symbolic_deliberation`: Core explicit symbolic step.
- Integrated into `tick()` with confidence gating and EMA calibration.

### v13.2 — External Symbolic + Self-Proposal + Phase C
- **Phase A**: `ExternalSymbolicInput` + hot-swappable external path
- **Phase B**: Mercy-gated `SymbolicSelfProposal` generation
- **Phase C**: Controlled `apply_symbolic_self_proposal`
- Features: `external-symbolic`, `self-proposal`, `experimental`

**Prefer v14.** Thunder locked in. yoi ⚡
