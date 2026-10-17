# patsagi-quantum-valence

Design-only names for Grover, QAOA, VQE, QFT, QPE, and surface-code QEC.

Council note: the algorithm claims are retired. The functions are not those algorithms.

- `valence_qaoa` returns a uniform vector of length `1 << p`. Each entry is `1/sqrt(2^p)`.
- `valence_vqe` returns `hamiltonian.trace() * valence_weight`. `ansatz_params` is not read.
- `valence_qft` returns a zero matrix.
- `valence_qpe` returns `0.0`. The arguments are not read.
- `valence_qec_surface` returns a copy of `logical_qubit` on both branches.
- `valence_grover_search` samples random bits. It is not Grover search.

Not a root workspace member. Standalone `[workspace]` with `resolver = "2"`.

AG-SML v1.2
