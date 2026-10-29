# Archived copies are not built

Council note (STUB-AUDIT-1 ARCHIVE-LINEAGE-RETIRE-1).

Everything under `docs/archive/` is an archived copy kept for lineage and history.

- These files are not compiled.
- They are not root members.
- There is no `Cargo.lock` under `docs/archive/`.
- `git ls-files 'docs/archive/**/Cargo.toml'` at `dd501b66209780f088dff651293c2eb1427d3e5d` counted 98 `Cargo.toml` files.
- `git ls-files 'docs/archive/**/*.rs'` at that same tip counted 396 `.rs` files.

Section 15 of `docs/STUB_AUDIT_1.md` records that 76 of those manifests fail loading the archived nexi workspace at `docs/archive/root-dirs/lineage/NEXi/Cargo.toml`.

Other one-off errors listed in that section:

- duplicate dependency name (`deeper_gadgets`)
- missing `[package]` (`mercy_orchestrator` copies)
- `no targets` (`legacy_fenca`, `docs/archive/root-dirs/crate-dupes/mercy`, `docs/archive/root-dirs/app-dumps/sdk/rust`)
- membership error only: `docs/archive/root-dirs/powrush/powrush`, `powrush-mmo-simulator`, `powrush_rbe_engine`, `powrush_faction_dynamics`, `powrush_particle_shaders`, plus lineage `self-evolution`, `xai-grok-bridge`, `symbiosis-layer`

Do not copy these files back into `crates/`. Do not add them to root `members`. Do not make them build.

Any numbers, valence literals, or superintelligence / AGSi / godly / eternal / operational wording inside archived files are historical source text, not results.
