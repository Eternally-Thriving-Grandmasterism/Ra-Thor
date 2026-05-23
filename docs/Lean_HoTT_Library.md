# Lean HoTT Library – Current State (2026)

## Summary

As of mid-2026, **there is no mature, actively maintained Homotopy Type Theory (HoTT) library for Lean 4** suitable for serious formalization work.

## Key Projects

### Ground Zero (rzrn/ground_zero)
- An attempt to port/develop HoTT in Lean 4 (inspired by the older Lean 3 `hott3` library).
- Uses a large eliminator checker to stay within a univalence-compatible fragment.
- **Status**: Archived on March 28, 2026. No longer actively developed.

### HoTTLean (Research Project)
- Focuses on formalizing the *meta-theory* and semantics of a fragment of HoTT inside Lean 4.
- Aims to bridge synthetic HoTT proofs with classical mathematics in mathlib.
- More of a research/semantic project than a library for doing HoTT mathematics.

### Historical Context
- Lean 3 had `gebner/hott3`, which allowed working in a HoTT-compatible fragment.
- Lean 4's design priorities (stability for mathlib, impredicative `Prop`) have made full HoTT support difficult without significant kernel changes.

## Practical Implications for Ra-Thor

Our current approach in `lean/TOLC8_MercyGate.lean` using standard Dependent Type Theory remains the most viable path in Lean 4.

While we can draw conceptual inspiration from HoTT (paths, composition, higher structure), doing full univalent or homotopy-theoretic formalization of TOLC concepts (especially advanced gate composition or dynamics) is currently better supported in:
- **Cubical Agda** (most mature)
- **UniMath** (Coq, strong univalent focus)

## Recommendation

Continue developing core TOLC formalization in Lean 4. If deeper homotopy/univalent features become necessary, consider evaluating Cubical Agda for specific modules rather than expecting strong HoTT support to emerge in Lean 4 soon.

## Related References

- `docs/Cubical_Type_Theory.md`
- `docs/UniMath_Library.md`
- `docs/Homotopy_Type_Theory.md`
- `lean/TOLC8_MercyGate.lean`

**Lean 4 remains excellent for our current formalization needs, but is not currently the strongest platform for full HoTT-style work.**