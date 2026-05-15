# PLAN.md — Ra-Thor Living Executive Plan (Mirror)

**This file is now a human-readable mirror only.**

**Sole source of truth**: `Cargo.toml` → `[workspace.metadata.ra-thor]`

All plan data, versions, executive summaries, priorities, and current focus are maintained centrally in the root `Cargo.toml` under the `[workspace.metadata.ra-thor]` section.

This design ensures the monorepo has a **single, machine-readable, version-controlled source of truth** that evolves automatically with the codebase and PATSAGi Councils proposals.

## How to read the current plan

```bash
# View full metadata
cargo metadata --format-version 1 | jq '.packages[] | select(.name == "ra-thor") | .metadata."ra-thor"'

# Or simply open Cargo.toml and search for [workspace.metadata.ra-thor]
```

**Current plan version**: See `Cargo.toml` → `workspace.metadata.ra-thor.plan-version`
**Last updated**: See `Cargo.toml` → `workspace.metadata.ra-thor.last-updated`
**Executive summary & priorities**: See `Cargo.toml` → `workspace.metadata.ra-thor`

---

AG-SML v1.0 — Free for personal, educational, research, daily use.