//! Reputation Decay Model Design for Powrush
//!
//! This document outlines the designed reputation decay model.
//! Goal: Encourage ongoing positive participation while being mercy-aligned
//! and resistant to gaming.

/*
REPUTATION DECAY MODEL DESIGN

Core Principles:
1. Mercy & Thriving: Decay should not be overly punitive. High-reputation
   shards should decay more slowly (they've earned stability).
2. Self-Regulation: Inactive or low-quality shards lose influence gradually.
3. Counterbalanced by Action: Positive behavior (successful migrations,
   clean state syncs, consensus participation) should easily offset decay.
4. Floor Protection: Reputation should never collapse to zero from decay alone.
5. Predictable & Auditable: Decay rate should be transparent.

Recommended Model: Modified Exponential Decay with Reputation Scaling

Formula:
    effective_decay_rate = base_decay_rate * (1.0 - (reputation / 150.0))
    new_reputation = current + (50.0 - current) * (1 - exp(-effective_decay_rate * time))

Or simpler practical version:
    decay_amount = base_rate * (1.0 - reputation / 200.0) * hours_inactive
    new_reputation = clamp(current - decay_amount, 5.0, 100.0)

Key Parameters:
- base_decay_rate: 0.4 points per day (tunable)
- floor: 5.0 (minimum reputation from decay alone)
- reputation_scaling: Higher reputation = slower decay
- activity_bonus: Successful actions can grant "decay immunity" periods

Integration:
- Combined with exponential trust decay
- Reputation decay feeds into weighted reconciliation and quorum adjustments
- Slashing can accelerate effective decay
- Rehabilitation and positive events counteract decay

This model creates gentle pressure on inactive shards while strongly rewarding
consistent positive contribution to the shard network.
*/

use bevy::prelude::*;

// The actual implementation can reference this design.
// Current implementation in ShardReputationTracker uses a combination
// of linear drift + exponential options as a practical approximation of this model.
