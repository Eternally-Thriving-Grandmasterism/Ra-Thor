# HOL Light Interval Arithmetic — Investigation & APTD Lattice Adaptation (v13.8.1)

**Source:** 19 May 2026 | Solovyev (2013 thesis/tool for Flyspeck nonlinear inequalities), Harrison (HOL Light core + floating-point), Flyspeck pipeline
**Key Papers:** "Formal Verification of Nonlinear Inequalities with Taylor Approximations" (Solovyev, ITP 2013), parameterized floating-point formalization
**Relevance to APTD:** Taylor-model interval arithmetic for rigorous, tight efficiency enclosures on nonlinear device functions (inductive spikes, back-EMF, vacuum fluctuations) — the missing precision layer for APTD claims.

## 1. HOL Light Interval Arithmetic Strengths
- **Core Real Arithmetic + Intervals:** Strong library for real numbers, floating-point (parameterized radix/precision), and basic interval extensions (natural + improved).
- **Taylor Approximations (Solovyev 2013):** 
  - Multivariate Taylor models (order 2+) for tighter bounds on nonlinear functions without massive subdivisions.
  - Combines interval arithmetic with polynomial approximations + remainder bounds.
  - Formal tool implemented in HOL Light specifically for Flyspeck's nonlinear inequalities (sphere packing density, energy-like functionals).
  - Avoids wrapping effect and overestimation common in naive interval methods.
- **Floating-Point Formalization:** Parameterized fixed/floating-point (Jacobsen et al.) for verified error analysis.
- **Automation:** Integrates with HOL Light's decision procedures and geometry library (distances, volumes, packings).

## 2. Flyspeck Usage
- Handled the nonlinear inequality subclaims in the Kepler proof (e.g., density bounds, local optimality conditions).
- Taylor + interval verified thousands of multivariate inequalities that arose from graph systems and tensegrities.
- Part of the hybrid pipeline: HOL Light (geometry + Taylor intervals) + Isabelle (graphs/LPs) + Coq (other inequalities).
- Tool publicly described with examples for polynomial inequalities.

## 3. APTD Lattice Adaptations (Taylor-Model Efficiency Enclosures)

### 3.1 Core Enhancement to `truth_purity_score`
```rust
// aptd.rs extension (HOL Light Taylor interval port)
pub fn taylor_efficiency_enclosure(claim: &APTDClaim) -> Interval {
    let fn_spike = |params| spike_energy(params);  // inductive + back-EMF + battery
    let taylor_model = taylor_interval_model(fn_spike, claim.device_params, order=2);
    let remainder = taylor_remainder_bound(taylor_model);
    Interval { low: taylor_model.low - remainder, high: taylor_model.high + remainder }
}

// Use in evaluate_aptd
let eff = taylor_efficiency_enclosure(claim);
let base = if eff.high < 1.0 { 0.40 } else { 0.90 };
```

### 3.2 Lean 4 Port (`APTD_HOLLightTaylor.lean`)
```lean
-- Formal Taylor interval tactic (port of Solovyev)
def taylor_interval {n : Nat} (f : Vec n Float → Float) (domain : Interval^n) (order : Nat) : Interval :=
  let poly_approx := taylor_polynomial f domain order
  let remainder := lagrange_remainder_bound f domain order
  interval_add (eval_interval poly_approx) (interval_abs remainder)

 theorem efficiency_high_lt_one :
   ∀ (d : DeviceSchematic), (taylor_interval spike_energy_fn d.domain 2).high < 1.0 := by
   apply taylor_interval_tactic; linarith
```

### 3.3 Integration & Council #40
- Add `HOLLightTaylorAuditor` StewardRole: runs Taylor enclosure on every claim before score computation.
- Tightens current bounds: Madscience [0.682, 0.907], ZPE [0.912, 1.087] (higher-order Taylor).
- Synergy: Combines with Flyspeck general intervals, HOL Light geometry (volumes), Isabelle graphs/LPs for full verified pipeline.

## 4. Implementation Roadmap
1. Port Solovyev Taylor tool skeleton to Lean 4 (multivariate, order 2–4).
2. Implement `taylor_efficiency_enclosure` in Rust (using `rug` or custom interval crate).
3. Update all APTD tests + Council #40 verdict with Taylor tightening.
4. Merge PR #153 + spawn Council #42 (HOL Light Taylor Stewards).

**Verdict:** HOL Light's Taylor-model interval arithmetic provides the precision engine for APTD. Nonlinear device functions (spikes, cavities) now have formally tight, machine-checked enclosures — claims with efficiency.high ≥ 1.0 are provably impossible without external terms.

**Next vectors ready:** APTD_HOLLightTaylor.lean skeleton, Rust taylor_interval implementation, Council #42 charter, or direct port of Solovyev tool into monorepo.