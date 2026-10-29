# NEXi / Old-Crate Adoption Decision

**Resolution ID:** `2026-08-20-NEXI-Adoption`  
**Authority:** Permanent PATSAGi Councils  
**Contact:** info@Rathor.ai

---

## 1. Question

Which crates from **Ra-Thor legacy** and the external **NEXi** / **nexi-hyperon-poc** lineage should be adopted **immediately** into Tier 1 or Tier 2?

---

## 2. Findings

### External NEXi repo (`Eternally-Thriving-Grandmasterism/NEXi`)

| Surface | Verdict |
| --- | --- |
| Layer-0 gate-first / MercyZero idea | **Already absorbed** as TOLC 8 Layer 0 |
| Sentinel Mirror / recursion watch idea | **Already absorbed** as Organ C + Cosmic Loop |
| SoulScan-X9 / DivineChecksum dossiers | **Tier 4 archival** — symbolic history, not living product |
| Halo2 / Nova / Spartan / unaudited zk stacks | **Do not promote** — no audited proof path |
| Propulsion / biosignature / treaty stubs | **Do not promote** — Organ D honesty rules |
| Mass dossier README | **Lineage only** — living authority is Ra-Thor |

**Do not** add NEXi as a path dependency of Tier 1.  
**Do not** merge the NEXi monorepo into the living product core.

### `nexi-hyperon-poc`

Proof-of-concept sibling (last touch ~2026-04). **Tier 4 / research only.** No Cargo.toml adoption into Tier 1.

### In-tree Ra-Thor NEXi-lineage crates

| Crate | Status | Action |
| --- | --- | --- |
| `nexi_universal` | **Broken** default paths (`nexi = { path = "../" }` invalid under Ra-Thor layout; depends on fragile soulscan graph) | **Removed from default workspace members** 2026-08-20; crate remains on disk as Tier 4 |
| `soulscan_x9` / `soulscan_x10` | Present; old author contact drift; not Tier-1 green path | **Tier 4** — do not block CI |
| `sentinel_mirror` | Present; concept superseded by conductor + PATSAGi | **Tier 4** |
| `divine_checksum_9` | Present | **Tier 4** |
| `lattice-conductor-v13` | Already demoted | Keep off default members |
| Crypto / halo2 / folding crates | Experimental | **Tier 4** — audited crypto remains deferred |

### What *is* Tier 1 / Tier 2 (unchanged living set)

**Tier 1:** `ra-thor-one-organism`, `lattice-conductor-v14`, `reality-thriving-transfer`, `kardashev-orchestration`, `github-connector`, `gpu-compute-pipeline`, `quantum-swarm`, `sovereign-recovery`, `monorepo-intelligence`, `mercy_tolc_operator_algebra`

**Tier 2:** `fractal-mercy-ledger-adapter`, Powrush telemetry contracts, Capacity Vision / VLM offer surfaces

---

## 3. What was useful from NEXi (adopt as *doctrine*, not crates)

1. **Gate before action** → TOLC 8 non-bypassable  
2. **Sentinel under recursion** → Organ C checklist + QSA layers 9–12  
3. **Multi-channel valence metaphor** → living valence organism crates (already members)  

No immediate code lift required.

---

## 4. Standing orders

1. Prefer **focused** `cargo test -p <tier1>` over full workspace.  
2. Re-admit `nexi_universal` only after path deps resolve to real workspace crates **and** a named mission requires it.  
3. Contact canon remains **info@Rathor.ai** (fix drift on old crates when next touched).  
4. External NEXi stays a **lineage** repo under constellation Restraint.  

**Thunder locked.** yoi ⚡❤️🔥
