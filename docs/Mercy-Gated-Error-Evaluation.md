# SovereignHealthMonitor + Mercy Integration (Production Examples)

## Overview

The mercy system is now wired into `SovereignHealthMonitor`. You can use mercy evaluation during health checks and blessing requests.

## Example 1: Run Sovereign Check with Mercy

```rust
use self_evolution::{init_sovereign_health_monitor, mercy_gating};

let mut monitor = init_sovereign_health_monitor();

// Run normal check (includes light mercy influence)
let metrics = monitor.run_sovereign_check();

// Or run with stronger mercy evaluation
let strong_metrics = monitor.run_sovereign_check_with_mercy();
```

## Example 2: Evaluate Current State

```rust
let verdict = monitor.evaluate_current_state_mercy(mercy_gating::MercyGateLevel::Integrative);

match verdict {
    mercy_gating::MercyVerdict::RequiresCouncilReview => {
        println!("Current state may need council attention");
    }
    _ => println!("State is within acceptable mercy parameters"),
}
```

## Example 3: Blessing with Mercy Synergy

```rust
let result = monitor.request_epigenetic_blessing("Important evolution step", true);

if result.success {
    println!("Blessing granted with mercy synergy. Tier: {:?}", result.tier);
}
```

AG-SML v1.0