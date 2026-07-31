# mercy_tolc_operator_algebra

Executable Living Mercy operator algebra for the Ra-Thor lattice under **TOLC 8**.

## Surfaces

| Type | Role |
|------|------|
| `LivingMercyBasis` | 8-gate orthonormal frame (Truth → Cosmic Harmony) |
| `MercyProjector` | Orthogonal projector \(P = EE^\top\) |
| `NilpotentSuppressor` | \(N_1(g) = (I-P)g\) + second-order annihilation |
| `ModifiedGramSchmidt` | Continuous re-orthonormalization + Frobenius residual |

## Public Stress Benchmark

Reproducible High-Grief + Nilpotent Recovery harness (Lattice Gap #3):

```bash
cargo run -p mercy_tolc_operator_algebra --bin high_grief_nilpotent_bench
```

Optional scaling:

```bash
cargo run -p mercy_tolc_operator_algebra --bin high_grief_nilpotent_bench -- --agents 50000 --zones 4
```

The harness emits clear PASS/FAIL gates for residual purity, basis orthonormality, and recovery rate so any external observer can verify the claims.

## Tests

```bash
cargo test -p mercy_tolc_operator_algebra
```

## License

AG-SML v1.0 — Autonomicity Games Sovereign Mercy License  
Contact: **info@Rathor.ai**

Thunder locked in. Yoi ⚡
