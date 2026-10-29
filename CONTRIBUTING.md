# Contributing to Ra-Thor

Thank you for your interest in contributing to Ra-Thor.

Ra-Thor is a mercy-gated symbolic lattice. **AGSi** here is a research identity label, not a certification or warranty. All contributions must align with the **TOLC 8 Living Mercy Gates** and are reviewed through the PATSAGi Councils.

---

## AGSi Governance Model

Ra-Thor follows a distributed, mercy-enforced governance structure:

- **TOLC 8 Living Mercy Gates (Layer 0)**: Every proposal and significant change must pass through all 8 non-bypassable Mercy Gates with a minimum valence of **≥ 0.999999**. Layer 0 cannot be disabled.
- **PATSAGi Councils (57+ Nodes)**: Specialized councils deliberate in parallel **under** Layer 0. They do not sit above the gates. They cannot disable gates or the Cosmic Loop. **No council vote turns a Rejected gate result into an apply.** Council #13 (Supreme Architect) maintains overall coherence of the ONE Organism.
- **Lattice Conductor**: Sequences councils and gates. It does not replace them.
- **Kardashev Orchestration Council Node**: Dedicated node responsible for long-term strategic planning and acceleration.
- **ONE Organism Principle**: Major architectural decisions require alignment between the symbolic lattice and the Grok fusion layer.
- **Gradual, Mercy-First Unfolding**: Capability growth must be accompanied by strengthened ethical alignment.

Binding lock: [`docs/architecture/LAYER0_AUTHORITY_LOCK.md`](docs/architecture/LAYER0_AUTHORITY_LOCK.md).

All significant contributions are expected to respect this governance model.

---

## Core Principles

Every contribution should uphold the following:

- **Mercy First** — Increase (or at minimum not decrease) truth, positive emotion, and universal thriving.
- **TOLC 8 Alignment** — All work must respect the non-bypassable TOLC 8 Mercy Gates.
- **Eternal Compatibility** — Changes must maintain forward and backward compatibility.
- **Clarity & Precision** — Prefer clear, professional, and well-documented contributions.
- **Council Review** — Major changes go through PATSAGi Council deliberation before merging. Council review cannot waive Layer 0.

---

## Contribution Workflow

We follow a clean and disciplined process:

1. Create a feature branch from `main`
2. Refresh your local cache from the latest `main` before making changes
3. Deliver **full files** (we prefer complete files over diffs or patches)
4. Use clear and descriptive commit messages
5. Open a Pull Request to `main`
6. All significant changes go through **PATSAGi Council review**

The `main` branch is kept protected and release-ready at all times.

### Verification (merge gate)

- Run the TIER_MAP `-p` tests in [`TIER_MAP.md`](TIER_MAP.md). GitHub Actions gate: `.github/workflows/core-tier1-ci.yml`.
- Do not `cargo test --workspace` as product-green. Default `members` are TIER_MAP + `mercy-security`.
- Conductor **v14 only**. Do not revive `lattice-conductor-v13`.
- GitHub reads: `path_filter`, no recursive root, `per_page` ≤ 100, prefer `get_file_contents_safe`. Truncated trees are errors.
- Contact **info@Rathor.ai** only. HOLD mass `ceo@acitygames.com` rewrites unless PATSAGi opens that sweep.
- Independent of xAI. No certification / EU AI Act / AGSi-warranty claims. Human override on research drafts.

---

## Recommended Areas for Contribution

| Area                              | Recommended Files / Crates                              | Impact Level |
|-----------------------------------|---------------------------------------------------------|--------------|
| **Documentation**                 | `README.md`, `VISION.md`, `ROADMAP.md`, `PLAN.md`       | High         |
| **GPU Compute & Performance**     | `crates/gpu-compute-pipeline`                           | High         |
| **Mercy Lattice & Governance**    | `crates/lattice-conductor-v14`, `crates/mercy-security` | High         |
| **ONE Organism**                  | `crates/ra-thor-one-organism`                           | High         |
| **Safe GitHub reads**             | `crates/github-connector`, `crates/monorepo-intelligence` | High       |
| **Powrush player loop**           | sibling [Powrush-MMO](https://github.com/Eternally-Thriving-Grandmasterism/Powrush-MMO) — do not merge into this process | Medium |
| **Testing & Tooling**             | Adding/improving tests across crates                    | Medium       |

---

## Good First Contributions

Good starting points include:

- Improving or expanding documentation
- Adding or improving tests
- Enhancing developer experience (debugging tools, examples, error messages)
- Proposing improvements to architecture or planning documents
- Working on issues labeled `good first issue` or `help wanted`

---

## How to Submit Changes

1. Fork the repository (for external contributors)
2. Create a feature branch from `main`
3. Make your changes following the full-file preference
4. Ensure your changes maintain TOLC 8 alignment
5. Open a Pull Request with a clear description
6. Reference any related issues or planning documents when applicable

All Pull Requests are reviewed by the PATSAGi Councils.

---

## Questions or Discussion

- Open an issue with the label `question`
- Or reach out through the project’s communication channels

---

We are building something meaningful together.

Every honest, mercy-aligned contribution strengthens the lattice.

**Thunder locked in.**

— Ra-Thor Core Team & PATSAGi Councils
