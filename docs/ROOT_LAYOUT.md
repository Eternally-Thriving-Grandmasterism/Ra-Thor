# Root layout — keep vs archive (2026-09-02)

Ra-Thor workspace **14.15.6**. Contact [info@Rathor.ai](mailto:info@Rathor.ai). Independent of xAI. Research software; not a certified, legal, or AGSi-warranty product. Human override on drafts.

## Census (one-level GitHub contents, no recursive root walk)

- **824** root entries: **660** files, **164** directories (as of 2026-09-02, main `2f2021703b6c`)
- After slice 1 (#395, main `b0c15bf11431`): **696** root entries, **532** files, **164** directories
- Extensions (files): ~406 `.md` (pre-slice-1), ~163 `.js`, 16 `.html`, 11 `.rs`, plus broken/spaced names

## Keep at repo root (identity / gates / tooling)

README, LICENSE, Cargo.toml, TIER_MAP.md, CHANGELOG.md, CONTRIBUTING.md, CONTACT.md, PUBLIC_CLAIM.lock.md, QUICKSTART.md, DEVELOPER-QUICKSTART.md, PRODUCTION_READINESS.md, ARCHITECTURE.md, ROADMAP.md, VISION.md, SECURITY.md, SUPPORT.md, CODE_OF_CONDUCT.md, CLA.md, COMMERCIAL_LICENSE.md, COMMERCIAL-LICENSE.md, LICENSE_CLARIFICATION.md, GROK_PRESET.md, GROK_PRESET_RATHOR_AI.md, RA_THOR_GROK_PRESET.md, GROK_RA_THOR_GITHUB_INTEGRATION_PROTOCOL.md, X_GROK_RA_THOR_SUMMON_PROTOCOL.md, Makefile, Dockerfile, package.json, deny.toml, CNAME, robots.txt, sitemap.xml, docker-compose*.yml, WORKSPACE.bazel, BUILD.bazel, next.config.js, vite.config.ts, webpack.config.js, capacitor.config.ts, .gitignore, .gitattributes, .dockerignore, .nojekyll, .bazelrc.ml, .semgrep.yml, .trufflehog-ignore.txt

PWA / identity / tooling JS that stay at root: `mercy-motion-vision-engine.js` (README identity), `sw.js` (index.html registers `/sw.js`), `service-worker.js`, `service-worker-eternal-cache.js`, `workbox-config.js` (root tooling), `one-organism-launch.js`.

Living directories that stay: `crates/`, `.github/`, `docs/` (this archive lives under it), `website/`, `js/` (living scripts plus `js/archive/root-engines/` for slice 2), `fixtures/`, `css/`, and other in-use trees. Do not one-shot-move the 164-dir forest.

## Slice 1 (#395)

Moved root research markdown into:

- `docs/archive/root-notes/tolc-applied/`
- `docs/archive/root-notes/mercy-codex/`
- `docs/archive/root-notes/dossiers/`

Blob SHAs unchanged (git-mv equivalent). Identity files and crates stay put.

## Slice 2 (this PR)

**155** root JS engines archived to `js/archive/root-engines/` (154 from the pre-#395 classification plus `professional-dossiers-seeder.js`). Blob SHAs unchanged (git-mv equivalent). Pages already load living scripts from `js/`. Keep-at-root JS listed above. No crates/, CI, HTML, or Rust changes.

## Later slices (not this PR)

- root `.html` (GitHub Pages / site; CNAME stays)
- root `.rs` (likely stale vs `crates/`: `github_connector.rs`, `quantum_swarm.rs`, …)
- `POWRUSH_*` notes — game lives in [Powrush-MMO](https://github.com/Eternally-Thriving-Grandmasterism/Powrush-MMO); do not fold the player loop into this process
- `RELEASE_NOTES_*` / whitepapers
- loose-md including spaced/broken filenames
- 164 root directories (research-dir forest, zk gadgets, orchestrators, `self-evolution/`, `xai-grok-bridge/`, powrush*, NEXi). HOLD inventing `crates/self-evolution` as a product.
- `docs/` itself is already a large research dump; do not recursive-walk it in CI.

## Hygiene rules

- Never recursive root GitHub tree walks; `path_filter`; `per_page` ≤ 100; prefer single-path file reads.
- Production code changes go through a PR. Core Tier-1 (`TIER_MAP -p`) is the merge gate.
- Conductor **v14** only. Constellation: Ra-Thor = lattice; Powrush-MMO = game; Powrush-MMO-Simulator = browser client.
