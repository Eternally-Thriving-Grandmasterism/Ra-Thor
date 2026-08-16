# soft-sovereign-agency

**Soft Sovereign Agency Layer** — Phase F of the Living Valence Organism  
Pure presentation and policy layer. Guidance is always invitation. Full sovereignty is preserved for every human and AI participant.

**Contact:** info@Rathor.ai  
**License:** AG-SML v1.0  
**Status:** Soft, feature-flaggable, TOLC 8 gated (default off)

## Purpose

Allows every participant to choose (and fluidly switch between) poetic/sensory or structured/mathematical views of the same underlying valence data. No authority or persistence model is changed.

## Feature Flag

`soft_sovereign_agency` (default **off**)

Activation requires all prior Living Valence Organism surfaces stable + dual-repo health verified.

## Core API

```rust
use soft_sovereign_agency::*;

let mut agency = SoftSovereignAgency::new("participant-1", Substrate::Human);
agency.set_view_mode(ViewMode::Structured);
