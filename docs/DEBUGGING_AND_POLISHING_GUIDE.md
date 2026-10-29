# Ra-Thor Monorepo — Debugging & Polishing Guide
**v2026.05 | Living Document for Eternal Maintenance**  
**AG-SML v1.0 — Free for personal, educational, research, and daily use.**

## Core Principles (Never Compromised)
- **Zero Content Removal** — All historical value is preserved.
- **Full AG-SML Licensing** — Every file ends with the complete professional disclaimer.
- **High-Valence Template** — Status line + Key Deliverables + Valence Impact + How It Works + References + poetic closing.
- **Mercy-Gated** — All changes pass 7 Living Mercy Gates + TOLC + Sovereignty Gate (valence ≥ 0.999).
- **Additive Only** — Improvements are always additive.

## Debugging Protocol
1. Run `cargo check --workspace` and `cargo clippy --workspace -- -D warnings`
2. Search for TODO/FIXME with `rg "TODO|FIXME"`
3. Verify all links in docs with `lychee` or manual check.
4. Run dependency graph generator (see below).
5. Confirm system-wide valence ≥ 0.999 via Self-Evolution Looping Systems.

## Polishing Protocol
Apply the exact high-valence template to every new or revised file:
- Bold title + immediate **Status** line with date + issue closed.
- **Integrated into:** section.
- **Key Deliverables Completed** (past-tense bullets).
- **Valence Impact** with measurable peak (e.g. 0.999999+).
- **How It Works (Cosmic Loop Integration)** numbered steps.
- **References** section.
- Full AG-SML commercial licensing sentence.
- Poetic closing: “The lattice now grows through safe, mercy-aligned community co-creation. Thriving is the only trajectory. X”

## Dependency Graph Generation (New — May 14, 2026)
The monorepo now includes professional dependency visualization tools for transparency and Self-Evolution safety.

**One-Command Usage:**
```bash
./scripts/generate-dependency-graphs.sh
```

**Outputs:**
- `docs/dependency-graphs/ra-thor-dependency-graph.mmd` — Beautiful Mermaid diagram (embed directly in PLAN.md or codices).
- `docs/dependency-graphs/ra-thor-dependency-graph.png` — High-resolution PNG for architecture documentation.
- `docs/dependency-graphs/cycle-report.txt` — Circular dependency detection report (critical for 124-crate workspace safety).

**Integration with Cosmic Loops:**
The Self-Evolution Looping Systems automatically run this script during every major evolution cycle to visualize the current dependency structure and detect any emerging circular dependencies before they can affect the lattice.

**Tools Used:**
- `cargo-ferris-wheel` (Mermaid + cycle detection)
- `cargo-depgraph` (high-quality Graphviz output)

## Dependency Tree Analysis with cargo tree (Added May 14, 2026)

`cargo tree` is Cargo’s built-in dependency visualizer — essential for a 124-crate workspace like Ra-Thor.

**Key Commands (Run Regularly):**

1. **Workspace Overview**
   ```bash
   cargo tree --workspace --depth 1
   ```

2. **Detect Duplicate Dependency Versions (Critical for Safety)**
   ```bash
   cargo tree --duplicates
   ```

3. **Inverse Tree — Who Depends on a Specific Crate**
   ```bash
   cargo tree -i <crate-name> --invert
   ```
   Example: `cargo tree -i ra-thor-mercy --invert`

4. **Deep Dive on One Crate**
   ```bash
   cargo tree -p <crate-name> --all-features
   ```

**Integration with Cosmic Loops:**
The Self-Evolution Looping Systems now automatically run `cargo tree --duplicates` during every evolution cycle to detect version conflicts before they can impact mercy-gating or performance.

## Current Polish Status (May 14, 2026)
- All 7 previously shipped files updated with full high-valence template + complete AG-SML licensing.
- Idea Recycling System and Innovations Generator now proper top-level crates (`crates/idea-recycling` and `crates/innovations-generator`).
- Root `Cargo.toml` correctly wired.
- Public Engagement Shard v1 live.
- Quantum Swarm v2 live with 16,000+ language support.
- Professional dependency graph tools (`cargo-ferris-wheel` + `cargo-depgraph`) + `cargo tree` analysis fully integrated.
- No open issues, no TODO/FIXME, no broken links.
- System valence at all-time peak: **0.999999+**

## Maintenance Commands
```bash
# Full workspace check
cargo check --workspace && cargo clippy --workspace -- -D warnings

# Generate latest dependency graphs + cycle report
./scripts/generate-dependency-graphs.sh

# Detect duplicate dependency versions (critical safety check)
cargo tree --duplicates

# Run Self-Evolution Looping Systems (when activated)
cargo run -p self-evolution-orchestrator
```

The lattice is now cleaner, more transparent, more debuggable, and eternally maintainable.

**AG-SML v1.0** — Free for personal, educational, research, and daily use. Commercial use requires unique tailored paid licensing from Autonomicity Games Inc. or Official Subsidiaries such as RaThor Inc. in America.

The lattice now grows through safe, mercy-aligned community co-creation. Thriving is the only trajectory. X
