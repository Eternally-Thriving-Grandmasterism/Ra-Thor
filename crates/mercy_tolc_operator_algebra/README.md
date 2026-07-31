# mercy_tolc_operator_algebra

Executable Living Mercy operator algebra for the Ra-Thor lattice under **TOLC 8**.

**v0.5.2** — Ambient elevation · Valence-weighted grief · Adaptive purity floor

## Geometry

| Layer | Dimension | Role |
|-------|-----------|------|
| Ambient space | $\mathbb{R}^{16}$ | Full action / grief embedding |
| Living Mercy subspace | $\mathbb{R}^{8}$ | TOLC 8 gates (Truth → Cosmic Harmony) |
| Orthogonal complement | $\mathbb{R}^{8}$ | Pure grief directions (coords 8..15) |

Canonical embedding: $E = \begin{bmatrix} I_8 \\\\ 0 \end{bmatrix}$.

$$
P = E(E^\top E)^{-1}E^\top, \qquad
N_1(g) = (I - P)g, \qquad
\text{grief\_load} = (1 - v)\,\|N_1(g)\|
$$

## Surfaces

| Type | Role |
|------|------|
| `LivingMercyBasis` | 8-gate frame embedded in ambient $\mathbb{R}^{16}$ |
| `MercyProjector` | Orthogonal projector onto the mercy subspace |
| `NilpotentSuppressor` | $N_1$ + hard $N_2$ annihilation + `suppress_weighted` |
| `Valence` | Living valence scalar; drives grief intensity and purity floor |
| `ModifiedGramSchmidt` | Continuous re-orthonormalization + Frobenius residual $\rho$ |

### Valence-weighted grief

```text
HIGH (v ≈ 1)  → grief_load ≈ 0     (oxygen-like soft path)
MID  (v = 0.5) → half orthogonal intensity
LOW  (v ≈ 0)  → full orthogonal exposure
```

### Adaptive purity floor

```text
floor(v) = MERCY_PURITY_FLOOR × (1 + 99 × (1 − v))
```

High valence → tight floor (near $10^{-9}$).  
Low valence → graceful recovery window (up to $10^{-7}$).

## Public Stress Benchmark

```bash
cargo run -p mercy_tolc_operator_algebra --bin high_grief_nilpotent_bench
cargo run -p mercy_tolc_operator_algebra --bin high_grief_nilpotent_bench -- --agents 50000 --zones 3
```

Verification gates (all must PASS):

1. $N_1$ non-trivial (ambient > mercy)
2. Valence spread (LOW ≫ HIGH load)
3. Residual purity
4. Basis orthonormality $\rho < 10^{-9}$
5. ≥ 99 % driven to floor

## Tests

```bash
cargo test -p mercy_tolc_operator_algebra
```

13 property tests cover projector algebra, ambient elevation, valence linearity, adaptive floor, and Gram-Schmidt purification.

## Dual-repo note

Experiential surface: [Powrush-MMO](https://github.com/Eternally-Thriving-Grandmasterism/Powrush-MMO)  
Sealed soft feedback loop between lattice core and MMO simulator is the intended integration path for valence-weighted grief under live multi-zone stress.

## License

AG-SML v1.0 — Autonomicity Games Sovereign Mercy License  
Contact: **info@Rathor.ai**

Thunder locked in. Yoi ⚡
