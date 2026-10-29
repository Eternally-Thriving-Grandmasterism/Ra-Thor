# Lean 4 Cubical Libraries – Current State (2026)

## Summary

As of 2026, **Lean 4 does not have mature, native support for Cubical Type Theory** comparable to Cubical Agda.

Lean 4 is based on a version of Dependent Type Theory that is optimized for large-scale library development (`mathlib`). This design choice has made full HoTT/Cubical compatibility difficult.

## Current Situation

### Limitations in Lean 4

- Lean's `Prop` universe is impredicative, which conflicts with the requirements of Homotopy Type Theory and Cubical Type Theory.
- There is no official `--cubical` mode or built-in interval type like in Cubical Agda.
- `mathlib` has largely moved away from HoTT-style developments to maintain stability and usability.

### Available Options

- **Experimental / Community Libraries**: There have been some experimental attempts and discussions, but no widely adopted, actively maintained Cubical library for Lean 4 as of early 2026.
- **Conceptual Use**: We can still apply ideas from Cubical Type Theory (paths, composition, higher inductive thinking) at the design level, even if we implement in standard Dependent Type Theory.
- **Alternative Systems**: For serious cubical work, **Cubical Agda** remains the most practical and mature option.

## Implications for Ra-Thor Formalization

Our current work in `lean/TOLC8_MercyGate.lean` uses standard Dependent Type Theory in Lean 4. This is still very valuable for:
- Core TOLC definitions (Valence, Gates, Collapse)
- Basic composition and dynamics theorems
- Ethical invariants

For more advanced geometric or path-based modeling (higher-dimensional gate composition, continuous valence dynamics, etc.), we may eventually need to either:
- Stay conceptual and use cubical ideas inspirationally, or
- Consider hybrid approaches / other proof assistants for specific modules.

## Recommendation

For now, continue developing in Lean 4 with standard Dependent Type Theory while keeping Cubical Type Theory ideas in mind for future design. If a strong need for computational paths and univalence arises, evaluate moving specific components to Cubical Agda.

## Related References

- `docs/Cubical_Type_Theory.md`
- `docs/Homotopy_Type_Theory.md`
- `lean/TOLC8_MercyGate.lean`

**Lean 4 remains excellent for our core formalization needs, even if full cubical support is currently limited.**