# Grok Bot lattice path (workspace 14.15.6)

Operator path for a Grok Bot session working this repo with PATSAGi Councils, Lattice Conductor v14, `github-connector`, and `monorepo-intelligence`.

Not an xAI affiliation. Not a certification. Drafts need human override before any public claim. Contact: info@Rathor.ai.

## Read

- Prefer a single known path. Use `github-connector::GitHubConnector::get_file_contents_safe`.
- Tree walks: `get_tree_safe` with a `path_filter`. Never recursive root. `per_page` ≤ 100.
- `crates/monorepo-intelligence` is protocol guardianship, not a license to walk the whole tree.

## Change

- Touch Tier 1 first (`TIER_MAP.md`). Conductor is `lattice-conductor-v14` only (v13 is deprecated).
- PATSAGi Councils live in `crates/patsagi-councils` and `crates/kardashev-orchestration`. Treat council docs as research labels until tests say otherwise.
- Land production edits on a branch and PR. Do not paste-overwrite main from chat.
- Refresh from current `main` before editing. Merge useful prior logic; do not discard it.

## Test

Run the TIER_MAP `-p` list. Do not `cargo test --workspace` on a PR and do not claim workspace-green.

Default Actions gate: `.github/workflows/core-tier1-ci.yml`.  
Opt-in full workspace: `workflow_dispatch` on `.github/workflows/ci.yml` and `ra-thor-ci.yml`.

## Do not

- Bump identity to 15.x in this pass.
- Re-add `nexi_universal` as a default member or revive conductor v13.
- Treat `legal-lattice`, `mercy_predictive_policing`, or `mercy_shield_law_enforcement` as shipped products.
- Invent eval scores. If evidence is missing, list the gap.
