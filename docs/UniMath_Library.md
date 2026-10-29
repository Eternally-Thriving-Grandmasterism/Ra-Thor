# UniMath Library Status (as of May 2026)

## Current Status

**UniMath** remains an active project. It is now often referred to as a **Rocq** library (the new name for Coq). The main repository is at:

- https://github.com/UniMath/UniMath

The project continues to formalize substantial mathematics from a **univalent point of view**.

## Related Projects under UniMath

- **agda-unimath**: A community-driven univalent mathematics library in Agda. This is actively maintained and has gained significant traction. It benefits from Agda’s strong support for cubical methods.

## Relevance to Ra-Thor Formalization

### Strengths
- One of the most mature realizations of Univalent Foundations.
- Strong focus on formalizing mathematics where equivalent structures are identified.
- Good reference for how to structure large-scale univalent formalizations.

### Limitations for Our Workflow
- Primarily in **Rocq/Coq**, while we are working in **Lean 4**.
- Switching would require significant effort.
- The Agda version (`agda-unimath`) may be more interesting due to better cubical/HoTT support.

## Recommendation

UniMath (and especially `agda-unimath`) remains valuable as:
- A conceptual and architectural reference.
- Inspiration for how to handle equivalence, identity, and higher structures in TOLC formalization.

For our immediate needs, continuing in Lean 4 with standard Dependent Type Theory is still the most practical path. However, `agda-unimath` is worth monitoring or exploring if we want stronger univalent/cubical capabilities in the future.

## Related References

- `docs/Lean_HoTT_Library.md`
- `docs/Cubical_Type_Theory.md`
- `docs/Univalent_Foundations.md`
- `lean/TOLC8_MercyGate.lean`

**UniMath continues to be one of the leading efforts in univalent formalization, with its Agda counterpart gaining momentum.**