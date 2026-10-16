# hyperon-metta-pln

String-formatting bridge. `SovereignHyperonMeTTaPLNBridge::reason` prefixes a query and concatenates format strings from the atomspace, MeTTa, and PLN helpers.

`mercy_gate_audit` returns true for every query. This crate does not run a MeTTa interpreter or a PLN solver.

`reason` writes `valence` to the source literal `0.9999` (not a measured value).

`src/adaptive_valence.rs` is on disk and is not a module.

Not a root workspace member. Standalone `[workspace]` with `resolver = "2"`.

AG-SML v1.1
