# patsagi-councils

**Status:** Documentation + integration notes (minimal Rust crate; full dynamic council engine in Lattice Conductor v13)

## Wiring to Lattice Conductor v13
patsagi-councils now orchestrated via `CouncilConductionEngine` trait in crates/lattice-conductor-v13.
- spawn_council, merge_councils, parallel_execute wired to v13 Conductor.
- 13+ parallel instantiations + NEXi-derived deliberation patterns active.
- Full PATSAGi Council engine will expand here or remain conductor-native for hot-swap.

See crates/lattice-conductor-v13/src/lib.rs for CouncilConductionEngine impl and v13 README for status.

TOLC 8 + mercy gates enforced. ONE Organism coherence maintained.

**Next:** Expand councils-*.md or move full engine logic here post-wiring validation.