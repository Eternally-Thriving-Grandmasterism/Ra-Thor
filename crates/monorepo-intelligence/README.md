# ra-thor-monorepo-intelligence

Tier-1 protocol guardian for the Ra-Thor workspace (14.15.6). Package `ra-thor-monorepo-intelligence` **0.3.11**.

Inspectable research software. Optional Grok session. **Not** xAI-affiliated. **Not** certified. **Not** a legal product. Contact: info@Rathor.ai.

## Standing read protocol

GitHub first. Path-filtered single-file reads. No recursive root walks.

| Rule | Bound |
|------|--------|
| Prefer | Single-path file reads (`github-connector` `get_file_contents_safe` / `get_tree_safe`) |
| Refuse | Recursive root tree walks |
| Pagination | `per_page` ≤ 100 |
| Local inventory | `WalkDir` `max_depth` 10; skip `target/`, `.git/`, `node_modules/` |

This crate’s `GitHubClient` is list/search only. Production tree and file reads go through `crates/github-connector`.

## Library surface (`src/lib.rs`)

```rust
use ra_thor_monorepo_intelligence::MonorepoIntelligence;

let mi = MonorepoIntelligence::new(".");
let scan = mi.full_scan()?;
let _ = mi.analyze_inheritance();
let _ = mi.hierarchical_predictive_coding(0.0, 4)?;
```

Public modules: `config`, `github`, `health`, `inheritance`, `plugin`, `predictive_coding`, `report`, `scanner`, `search`.

Re-exports: `MonorepoScanner`, `ScanError`, `ScanResult`, `ScannedFile`, `InheritanceStatus`, `HierarchicalPredictiveCoding`, `PredictiveCodingError`, `PredictiveCodingResult`, `MERCY_VALENCE_FLOOR`.

## Optional CLI

Bin `ra-thor-monorepo-intelligence` is gated on `--features cli` (`src/main.rs`). Default `cargo test -p ra-thor-monorepo-intelligence` tests the **lib**, not the bin.

## Verify

```bash
cargo test -p ra-thor-monorepo-intelligence
```

Do not `cargo test --workspace` as a product-green gate. See `TIER_MAP.md`.

## Claim lock

Workspace cite stays **14.15.6**. Gap OPEN. No 15.x. No ninth gate. No certification / AGSi-as-product / legal-product / xAI endorsement claims.

License on this crate manifest: AG-SML v1.0 (workspace root currently states AG-SML v1.1).
