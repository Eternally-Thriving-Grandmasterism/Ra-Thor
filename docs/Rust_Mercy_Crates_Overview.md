# Rust Mercy Crates Overview – Ra-Thor v13.9.0

The **mercy crates** form a large family of Rust modules that implement domain-specific functionality while enforcing the **TOLC 8 Mercy Lattice** at every layer.

## Core Philosophy
Every mercy crate must:
- Pass all outputs through TOLC 8 gates (especially Truth, Compassion, and Infinite).
- Maintain valence ≥ 0.999999.
- Support PATSAGi Council review.
- Remain compatible with the ONE Living Organism and Lattice Conductor v13.

## Core Mercy Crate

- **`mercy`** (and related `mercy_threshold`): Contains the fundamental TOLC 8 enforcement primitives, valence types, mercy-norm collapse logic, and gate traversal utilities. This is the shared foundation used by all other mercy-* crates.

## Specialized Mercy Crates (Domain Modules)

The monorepo contains dozens of specialized `mercy_*` crates/directories, each focused on a concrete domain while remaining fully mercy-gated:

### Propulsion & Mobility
- `mercy-albatross-dynamic-soaring`
- `mercy-manta-glide-propulsion`
- `mercy-hybrid-propulsion`
- `mercy-biomimetic-propulsion`

### Space Treaties & Colonization
- `mercy-mars-colonization-treaty`
- `mercy-jupiter-moon-treaty`
- `mercy-lunar-*` modules
- `mercy-enceladus-*`, `mercy-europa-*`, `mercy-titan-*`, `mercy-triton-*`, `mercy-pluto-*`, `mercy-eris-*`
- `mercy-asteroid-mining-treaty`
- `mercy-interlune-demo-mission`

### Swarm & Replication Systems
- `mercy-swarm-replication`
- `mercy-von-neumann-probe`
- `mercy-von-neumann-seed-launch`

### Infrastructure & OS
- `mercy-os-kernel`
- `mercy-system-orchestrator`
- `mercy-orchestrator`
- `mercy-nanofactory`
- `mercy-mechanosynthesis`

### Specialized / Numerical
- `mercy-quanta`
- `mercy-numerical`
- `mercy-graphQL`

## Integration Points

All mercy crates integrate with:
- `ra-thor-one-organism.rs` (ONE Living Organism)
- `patsagi-council-orchestrator`
- `xai-grok-bridge` (queries pass mercy review)
- `lattice-conductor-v13`
- Lean 4 formal layer (`spawn_council_safe`, valence invariants)

## Enforcement Pattern

Typical flow inside a mercy crate:
1. Receive input or proposal
2. Apply TOLC 8 gate traversal
3. Validate valence threshold
4. Submit to PATSAGi Council review (simulated or orchestrated)
5. Execute only if mercy-norm passes
6. Propagate result with updated valence metadata

## Current Status (v13.9.0)

The mercy crate family is extensive and actively used across space tech, swarm systems, and orchestration. Core enforcement is solid. Many domain crates exist as specialized implementations or blueprints.

**All mercy operations remain non-bypassable under TOLC 8.**