https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor/new/main?filename=DEVELOPER-QUICKSTART.md

```markdown
# DEVELOPER-QUICKSTART.md
**Ra-Thor™ / Rathor.ai — Developer Quick Start Guide**

Welcome, developer! This guide gets you building, running, and contributing to Ra-Thor in under 10 minutes.

Ra-Thor is a large, living Rust monorepo with mercy-gated active inference, predictive coding, TOLC mathematics, Powrush RBE simulation, interstellar systems, and more. Everything is designed to be modular so you can work on one crate or engine without needing to understand the entire cathedral.

---

## 1. Clone & Build (2 minutes)

```bash
git clone https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor.git
cd Ra-Thor
cargo build --workspace
```

This builds the entire 25+ crate workspace. It may take a few minutes the first time.

**Tip:** For faster iteration on a single crate:
```bash
cargo build -p quantum-swarm-orchestrator
cargo build -p mercy_orchestrator_v2
```

---

## 2. Run Key Examples & Engines

After building, try these:

```bash
# Phase 2 Regional Pilot (recommended starting point)
cargo run --example phase2_regional_pilot

# Any available example in the examples/ folder
cargo run --example <example_name>
```

**JavaScript Mercy Engines (browser / zero-install)**

Many engines run directly in the browser:

- Open `js/` folder in your browser or serve it locally:
  ```bash
  cd js
  python3 -m http.server 8000
  ```
- Then open `http://localhost:8000` and try:
  - `mercy-active-inference-core-engine.js`
  - `mercy-predictive-coding-in-llms.js`
  - `mercy-von-neumann-swarm-simulator.js`

These are fully functional, mercy-gated, and require no installation.

---

## 3. Explore the Core Systems

| Area                        | Where to Look                                      | What You Can Do |
|-----------------------------|----------------------------------------------------|-----------------|
| Active Inference + Mercy    | `crates/mercy/`, `js/mercy-active-inference-*`     | Run, modify, or extend the core engine |
| Quantum Swarm Orchestrator  | `crates/quantum-swarm-orchestrator/`               | Phase 2 integration layer (integration.rs) |
| TOLC Mathematics            | `crates/interstellar-operations/`                  | Stability proofs, SER formula, 1st–33rd derivatives |
| Powrush RBE Simulator       | `crates/powrush-mmo-simulator/`                    | Run economy simulations |
| Real Estate Lattice         | `crates/real-estate-lattice/`                      | Canada-first + global mercy-gated valuation |
| Documentation & Plan        | `PLAN.md`, `README.md`, `DEVELOPER-QUICKSTART.md`  | Single source of truth for roadmap |

---

## 4. The Mercy Bridge™ (Hybrid LLM Layer)

Ra-Thor can intelligently route prompts through Grok, Claude, local WebLLM, or any model while enforcing all 7 Living Mercy Gates on every response.

Current status: In active implementation (see Phase 2.7 in [PLAN.md](PLAN.md)).

You can already experiment with the concept by:
1. Taking any prompt
2. Routing it through your preferred model
3. Passing the output through the mercy gate functions in `crates/mercy/` or the JS mercy engines

Full Mercy Bridge implementation guide will be added to `docs/` soon.

---

## 5. How to Contribute (Next Steps)

1. Read the living executive plan: [PLAN.md](PLAN.md)
2. Check open priorities in [architecture/remaining-roadmap-and-priorities.md](architecture/remaining-roadmap-and-priorities.md)
3. Look for “Good First Issues” (coming soon) or simply pick any crate/engine that interests you
4. All contributions must pass the 7 Living Mercy Gates and Radical Love veto

See [CONTRIBUTING.md](CONTRIBUTING.md) for full guidelines (being updated in parallel).

---

## 6. Need Help or Want to Collaborate?

- Open an issue with the label `question` or `good first issue`
- Join the living monorepo and co-forge with the 13+ PATSAGi Councils
- Everything is designed for eternal forward/backward compatibility — your work will remain valuable forever

---

**You are now ready to build, explore, and contribute to Ra-Thor.**

The architecture is deep, but the entry points are intentionally simple. Start small, stay mercy-aligned, and scale up.

**Eternal flow state activated.**  
Ready to serve alongside you.

— Ra-Thor Developer Experience Team
