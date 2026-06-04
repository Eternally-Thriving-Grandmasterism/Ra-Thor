/*!
# Powrush Particle Shaders — Batching Indirect Draw Calls

Advanced batching of indirect draw calls for efficient rendering of many particle effects.

## Why Batch Indirect Draws?
In a full Powrush MMO scene you may have dozens or hundreds of active particle systems
(different factions, council events, high-reputation bursts, environmental effects).

Issuing one draw call per effect is expensive.
Batching multiple `DrawIndirect` commands allows:
- Single (or very few) draw calls for many effects
- Better GPU utilization
- Reduced CPU overhead
*/

use powrush_faction_dynamics::{Faction, FactionVisualIdentity, ParticleParams};

#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct DrawIndirect {
    pub vertex_count: u32,
    pub instance_count: u32,
    pub first_vertex: u32,
    pub first_instance: u32,
}

/// Represents a batch of indirect draw commands.
/// Can be uploaded as a storage buffer and used with `multi_draw_indirect`.
#[derive(Debug, Clone)]
pub struct BatchedIndirectDraws {
    pub commands: Vec<DrawIndirect>,
}

impl BatchedIndirectDraws {
    pub fn new() -> Self {
        Self {
            commands: Vec::new(),
        }
    }

    pub fn add(&mut self, cmd: DrawIndirect) {
        self.commands.push(cmd);
    }

    pub fn len(&self) -> usize {
        self.commands.len()
    }

    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }

    /// Prepares batched indirect commands from multiple particle effects.
    /// In a real system this would come from a compute pass that generates
    /// one indirect command per visible effect/batch.
    pub fn from_effects(effects: &[ParticleEffect]) -> Self {
        let mut batch = Self::new();
        for effect in effects {
            let cmd = prepare_indirect_draw(effect.vertex_count_per_particle);
            // instance_count would normally be written by culling compute
            batch.add(cmd);
        }
        batch
    }
}

/// Helper to create a single indirect command (instance_count set later by culling).
pub fn prepare_indirect_draw(vertex_count_per_particle: u32) -> DrawIndirect {
    DrawIndirect {
        vertex_count: vertex_count_per_particle,
        instance_count: 0,
        first_vertex: 0,
        first_instance: 0,
    }
}

/// Example of how a compute shader can write multiple indirect commands.
/// In practice you would have one entry per particle system / batch.
pub mod compute {
    pub const BATCHED_CULLING_AND_INDIRECT: &str = r#"
        // This compute shader can output multiple DrawIndirect commands
        // (one per effect or spatial batch) and fill their instance_count.

        struct DrawIndirect {
            vertex_count: u32,
            instance_count: u32,
            first_vertex: u32,
            first_instance: u32,
        };

        @group(0) @binding(0) var<storage, read_write> indirect_commands: array<DrawIndirect>;
        @group(0) @binding(1) var<storage, read_write> visible_counts: array<atomic<u32>>;

        // ... culling logic per effect/batch ...
        // Then write:
        // indirect_commands[effect_index].instance_count = visible_count;
    "#;
}

pub mod wgsl {
    // ... existing shader snippets ...
}
