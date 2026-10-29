/*!
# Lattice Conductor — Self-Evolution Telemetry Module
# crates/lattice-conductor-v13/src/self_evolution_telemetry.rs

**Version**: v0.1  
**Date**: 2026-07-11
**License**: Autonomicity Games Sovereign Mercy License (AG-SML) v1.0

## Purpose
This module provides the Lattice Conductor with structured handling of
backend-aware self-evolution telemetry emitted by `MasterKernel`.

It allows the Conductor to:
- Record and store evolution events
- Query recent evolution history
- Feed telemetry into higher observability / PATSAGi Council systems

## Integration
```rust
use crate::kernel::master_kernel::SelfEvolutionTelemetry;
use crate::lattice_conductor_v13::self_evolution_telemetry::ConductorSelfEvolutionRecorder;

let recorder = ConductorSelfEvolutionRecorder::new();
if let Some(telemetry) = master_kernel.get_last_self_evolution_telemetry() {
    recorder.record(telemetry.clone());
}
```
*/

use crate::kernel::master_kernel::SelfEvolutionTelemetry;
use std::collections::VecDeque;

/// Maximum number of recent evolution events to keep in memory
const MAX_HISTORY: usize = 128;

/// Records and manages self-evolution telemetry for the Lattice Conductor
pub struct ConductorSelfEvolutionRecorder {
    history: VecDeque<SelfEvolutionTelemetry>,
}

impl ConductorSelfEvolutionRecorder {
    pub fn new() -> Self {
        Self {
            history: VecDeque::with_capacity(MAX_HISTORY),
        }
    }

    /// Record a new self-evolution event from MasterKernel
    pub fn record(&mut self, telemetry: SelfEvolutionTelemetry) {
        if self.history.len() >= MAX_HISTORY {
            self.history.pop_front();
        }
        self.history.push_back(telemetry);
    }

    /// Get the most recent evolution event
    pub fn latest(&self) -> Option<&SelfEvolutionTelemetry> {
        self.history.back()
    }

    /// Get all recorded events (newest last)
    pub fn all(&self) -> Vec<&SelfEvolutionTelemetry> {
        self.history.iter().collect()
    }

    /// Get events filtered by backend
    pub fn by_backend(&self, backend: crate::kernel::gpu_compute_pipeline::GpuBackend) -> Vec<&SelfEvolutionTelemetry> {
        self.history
            .iter()
            .filter(|t| t.backend == backend)
            .collect()
    }
}

impl Default for ConductorSelfEvolutionRecorder {
    fn default() -> Self {
        Self::new()
    }
}

/*!
## Usage in Lattice Conductor

This recorder can be embedded in the main `LatticeConductor` struct or used
as a standalone telemetry sink.

It provides the bridge between `MasterKernel`'s backend-aware self-evolution
and the Conductor's observability / governance layer.

Thunder locked in. Self-evolution telemetry is now native to the Lattice Conductor.
*/
