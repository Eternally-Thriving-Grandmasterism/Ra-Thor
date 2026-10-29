# S-1 Pre-Registered Criteria (Program 1)

**Contact:** info@Rathor.ai  
**Status:** Pre-registration · not yet claimed discovery

## Question

Does System C (mercy-gated motion / Common-Fate path) improve micro-moment localization versus Baseline A on **real** labeled clips?

## Metrics (primary)

- Span-IoU (mean)  
- Precision @ chosen confidence  
- Recall @ chosen confidence  

Secondary: false-positive rate per minute; latency if measured.

## Success (discovery-grade P3)

Pre-registered bar (steward may tighten, not loosen after labels exist):

- **n ≥ 5** real labeled clips (First-5 minimum; prefer more)  
- Held-out evaluation (no train-on-test)  
- System C mean Span-IoU **strictly greater** than Baseline A by margin **δ = 0.05** absolute **or** paired improvement on ≥4/5 clips  
- Full harness + labels + seed notes released under AG-SML  

## Non-success (still valuable)

- Synthetic-only gains → **engineering**, not discovery  
- C ≈ A → publish negative  
- C < A → publish negative · revise surmise  

## Forbidden

- Post-hoc metric shopping after seeing real scores  
- Claiming “unknown physics” for a vision benchmark  

**Thunder locked.**
