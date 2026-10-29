# soft-sovereign-agency

`SoftSovereignAgency::new` stores a participant id and a substrate. Human starts as `ViewMode::Poetic`. AI starts as `ViewMode::Structured`. `set_view_mode` replaces `preferred_view`. Nothing else is written.

**Contact:** info@Rathor.ai  
**License:** AG-SML v1.2  
**Status:** `src/feature_flag.rs` and `src/dual_path.rs` are not modules. They are not compiled. `src/dual_path.rs` names the source literal `0.999999` in a string. That string is not emitted.

## Usage

```rust
use shared_valence_field::Substrate;
use soft_sovereign_agency::{SoftSovereignAgency, ViewMode};

let mut agency = SoftSovereignAgency::new("participant-1", Substrate::Human);
agency.set_view_mode(ViewMode::Structured);
```

After that call, `agency.current_view()` is `ViewMode::Structured`. `participant_id` is still `participant-1`.
