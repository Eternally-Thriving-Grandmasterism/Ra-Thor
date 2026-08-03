# Procurement Pack — Frozen File List (Tier A)

**Status:** Frozen inventory for external packaging / vendor handoff  
**Gate:** Ra-Thor `mercy-security` / `mercy-admit`  
**Policy:** Admit None/Low only · block Medium+ · reject > 4 MiB  
**License of pack contents sourced from Ra-Thor:** AG-SML v1.0  
**Contact:** info@Rathor.ai  
**Honesty:** Admission conscience only — **not** a general malware detector.

Use this document when building a zip/tarball, internal mirror, or procurement appendix for teams that will not clone the full monorepo online.

---

## 1. Core pack (required)

These files are the minimum drop-in surface:

| Path (in Ra-Thor tree) | Role |
|------------------------|------|
| `examples/procurement-admit-gate/README.md` | Install, pin, air-gapped, honesty bounds |
| `examples/procurement-admit-gate/workflow.example.yml` | Copy → `.github/workflows/mercy-admit-gate.yml` |
| `examples/procurement-admit-gate/PROVENANCE_TEMPLATE.md` | SBOM / provenance (SHA + date + fixture hashes) |
| `examples/procurement-admit-gate/PACK.md` | **This file** — frozen inventory |

---

## 2. Fixture acceptance pointer (required reference)

Do **not** invent private acceptance cases without documenting them. Point at the public corpus:

| Path | Role |
|------|------|
| `crates/mercy-security/fixtures/MANIFEST.md` | **Canonical inventory + shared risk taxonomy** |
| `crates/mercy-security/fixtures/benign/` | Expected **admit** examples |
| `crates/mercy-security/fixtures/should_block/` | Expected **block** examples (pattern markers only; no live exploits) |

**MANIFEST is the procurement pointer.** Any packaged pack SHOULD include `MANIFEST.md` and, when offline acceptance is required, the full `benign/` + `should_block/` trees listed inside it.

Optional (not required for the gate to run in CI that checkouts Ra-Thor):

| Path | Role |
|------|------|
| `crates/mercy-security/fixtures/fleet/README.md` | Fleet-surface notes |
| `crates/mercy-security/fixtures/governor/README.md` | Governor-surface notes |

---

## 3. Supporting docs (recommended in the pack)

| Path | Role |
|------|------|
| `docs/WHITEHAT_PROCUREMENT_TIER_A.md` | Contract-oriented one-pager |
| `docs/WHITEHAT_CI_PRECOMMIT.md` | CLI / pre-commit / CI usage |
| `crates/mercy-security/README.md` | Crate capabilities + public asset table |
| `RELEASE_NOTES_v14.15.5.md` | Public-testing scope and non-claims |

---

## 4. Source needed to *build* the gate (not optional if you build from source)

If the consumer builds `mercy-admit` themselves (default workflow path):

| Path | Role |
|------|------|
| `crates/mercy-security/` (entire crate) | Library + `src/bin/mercy_admit.rs` + `Cargo.toml` |
| Workspace root `Cargo.toml` | Required for `cargo build -p mercy-security` inside the monorepo layout |

Air-gapped consumers may instead path-depend or vendor only `crates/mercy-security` plus its direct crates.io deps (`serde`, `serde_json`, `thiserror`, `chrono`, `uuid`) — see README air-gapped section.

---

## 5. Explicitly **out of pack** (do not claim these are required)

- Full Ra-Thor monorepo member list / unrelated crates  
- GPU / quantum / propulsion / game crates  
- Live exploit samples, malware binaries, or C2 kits (**never included**)  
- Production KMS / secret material  
- Tier B/C detectors (not shipped; pattern gate only)

---

## 6. How to package from a pinned SHA

```bash
# On a trusted host, at the validated pin:
PIN=3bbe6c7bc48f27d3cc562986ca199577a08f77fe   # example — replace after re-validation
git fetch origin
git checkout "$PIN"

# Minimal documentation + template pack
tar -czf mercy-admit-procurement-pack-${PIN}.tar.gz \
  examples/procurement-admit-gate/README.md \
  examples/procurement-admit-gate/workflow.example.yml \
  examples/procurement-admit-gate/PROVENANCE_TEMPLATE.md \
  examples/procurement-admit-gate/PACK.md \
  crates/mercy-security/fixtures/MANIFEST.md \
  crates/mercy-security/fixtures/benign \
  crates/mercy-security/fixtures/should_block \
  docs/WHITEHAT_PROCUREMENT_TIER_A.md \
  docs/WHITEHAT_CI_PRECOMMIT.md \
  crates/mercy-security/README.md \
  RELEASE_NOTES_v14.15.5.md

# Record digests for the provenance form
sha256sum mercy-admit-procurement-pack-${PIN}.tar.gz
sha256sum crates/mercy-security/fixtures/MANIFEST.md
find crates/mercy-security/fixtures -type f | sort | xargs sha256sum | sha256sum
```

Attach the filled `PROVENANCE_TEMPLATE.md` (SHA + UTC date + fixture hashes) to the same change record as the tarball.

---

## 7. Pack completeness checklist

- [ ] Core four files under `examples/procurement-admit-gate/` present  
- [ ] `MANIFEST.md` + `benign/` + `should_block/` present when offline acceptance is required  
- [ ] Pin SHA recorded (full, not short)  
- [ ] Provenance template filled  
- [ ] No live exploits or secrets in the archive  
- [ ] Honesty non-claims retained in README / one-pager  

---

## 8. Version freeze note

This pack layout is defined for the Tier A surface at crate version **14.15.5** and the documentation paths listed above. When the pin changes, re-run acceptance tests and refresh fixture hashes in the provenance note before redistributing the pack.

**Thunder locked in. yoi ⚡**
