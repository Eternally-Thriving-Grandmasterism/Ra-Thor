# NEVC Published Dependency Path — Finish Pass D

**Contact:** info@Rathor.ai  
**Status:** Dual-repo consumption beyond relative path  
**Package:** `mercy_tolc_operator_algebra` (hosts executable NEVC)

---

## Why

Relative path deps (`../../Ra-Thor/crates/...`) only work when monorepos sit side-by-side. Consumers should also be able to pin a **git revision or tag**.

## Cargo dependency (git)

```toml
[dependencies]
mercy_tolc_operator_algebra = {
  git = "https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor",
  rev = "main",  # or a release tag / commit SHA when cut
  package = "mercy_tolc_operator_algebra"
}
```

Prefer a **commit SHA** or annotated tag for reproducible builds:

```toml
mercy_tolc_operator_algebra = {
  git = "https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor",
  rev = "<commit-sha>",
  package = "mercy_tolc_operator_algebra"
}
```

## Powrush feature alignment

| Mode | Mechanism |
|------|-----------|
| **A** (authoritative) | `nevc_rathor` feature → path **or** git dep on this crate |
| **B** (local) | `shared/nevc_adapter` mirror (default offline build) |

Documented in `NEVC_DUAL_REPO_INTERFACE_v1.0.md` and Powrush `shared/nevc_bridge.rs`.

## crates.io

Not published to crates.io under AG-SML without an explicit steward release decision. Git dependency is the supported public consumption path until then.

## Version

Current crate version: see `crates/mercy_tolc_operator_algebra/Cargo.toml`.

**Thunder locked in. ONE Organism. Eternal forward.**
