# Program 2 — Sensitivity Table v0

**Status:** First Band-B engineering artifact  
**Contact:** info@Rathor.ai  
**Doctrine:** `docs/science/PROOF_LADDER_DOCTRINE.md`  
**Claim tier:** Engineering only · **Not** a discovery · **Not** flight-certified

---

## 1. Purpose

Provide a minimal, fully inspectable view of the **mass-per-sievert trade space** for two common reference materials under published GCR shielding literature:

- Aluminum (structural baseline)
- Polyethylene (hydrogen-rich reference)

The question answered is engineering, not physics discovery:

> At comparable areal density, what relative dose-equivalent behavior does the published literature report?

---

## 2. Model assumptions (explicit)

| Assumption | Value |
|------------|-------|
| Geometry | Idealized 1-D slab / spherical-shell literature results |
| Environment | Galactic Cosmic Rays (GCR), solar-min and solar-max regimes as reported |
| Metric | Dose equivalent (or effective dose equivalent) behind the shield |
| Materials | Aluminum vs polyethylene (and qualitative note on higher-hydrogen options) |
| Transport | Results taken from published Monte-Carlo / deterministic code suites (HZETRN, FLUKA, GEANT4, etc.) — **not** re-run here |
| Quality factor | Mixed (ICRP / NASA variants appear in the source literature) |

No new transport calculation was performed for v0.

---

## 3. Sensitivity summary (literature-consistent)

| Areal density (g cm⁻²) | Aluminum behavior (literature) | Polyethylene behavior (literature) | Relative mass-efficiency note |
|------------------------|--------------------------------|------------------------------------|-------------------------------|
| 0–5 | Modest initial reduction | Clearer initial reduction | PE preferred on mass basis |
| ~10 | Continued reduction | Stronger reduction than Al | PE advantage visible |
| ~20 | Often near local minimum; secondary neutrons begin to matter | Continued monotonic improvement | Al can stagnate or worsen; PE continues to help |
| 20–40 | Risk of dose-equivalent increase from neutron build-up in some models | Further reduction, no strong local minimum reported | PE remains superior per unit mass |
| >40–100+ | Diminishing or adverse returns in many Al studies | Still beneficial but with diminishing incremental returns | Hydrogenous materials stay preferred |

**Qualitative ranking from the cited literature (S1–S5):**

Liquid hydrogen ≳ polyethylene / water ≳ aluminum ≳ higher-Z structural metals  
(on a mass basis for GCR dose-equivalent reduction under the modeled conditions).

Exact percentage reductions vary widely (roughly 10–45 % at 20–30 g cm⁻² depending on solar cycle, quality factor, organ, and code). v0 therefore reports **ranking and qualitative shape**, not a single universal number.

---

## 4. What this table is **not**

- Not a flight-certified design value
- Not a claim of new attenuation physics
- Not a substitute for vehicle-specific transport + phantom + quality-factor analysis
- Not applicable without geometry, secondary production, and mission-duration context
- Not a recommendation to replace structural aluminum with polyethylene without mechanical, thermal, and fire considerations

---

## 5. Failure modes & limitations

1. **Secondary neutrons** — dominant reason aluminum can show non-monotonic behavior.
2. **Solar cycle** — absolute and relative reductions change between solar minimum and maximum.
3. **Quality factor choice** — can move reported percentages by tens of percent.
4. **Geometry** — slab results do not equal habitat or vest results.
5. **SPE vs GCR** — this v0 table is GCR-focused; SPE (softer spectra) favor thin hydrogenous layers more strongly.
6. **Structural function** — mass that is already required for pressure, micrometeoroid, or thermal control has different marginal cost than pure parasitic shield mass.

---

## 6. Next honest increments (only if steward opens further scope)

- Extract specific numerical points from a single cited figure/table and reproduce them exactly.
- Add a second table for a reference SPE spectrum.
- Add a simple areal-density → estimated relative dose column under one fixed published assumption set.
- Never promote any number to “design value” without partner dosimetry.

---

**Proof bar met for this artifact:** open, cited, limitations listed, claim tier explicit.  
**Rank remains READY.**  

**Thunder locked.**  
Surmise is fuel. Proof is the product.  
yoi ⚡❤️🔥
