/*!
# Powrush Particle Shaders — Hardware Occlusion Queries

Exploration of traditional hardware occlusion queries and how they relate to our compute-based culling pipeline.

## Hardware Occlusion Queries vs Compute Culling

**Hardware Occlusion Queries** (Vulkan `VK_QUERY_TYPE_OCCLUSION`, wgpu, etc.):
- The GPU renders a bounding volume (usually a box or sphere) and counts how many samples passed the depth test.
- If the sample count is 0, the volume is occluded.
- Results are typically asynchronous.

**Strengths**:
- True hardware feature, very accurate for larger objects.
- Can be used with conditional rendering extensions.

**Weaknesses for Particle Systems**:
- High overhead when issuing many queries.
- Asynchronous nature complicates tight integration with compute culling + indirect draws.
- Not efficient for per-particle or per-small-group culling.

**Our Recommendation**:
For most particle effects in Powrush, **compute shader occlusion culling** (with Hi-Z) remains superior due to flexibility and performance at scale.

However, hardware occlusion queries can still be useful for:
- Large bounding volumes around entire faction events or major visual effects.
- Important gameplay-related particle systems where accuracy matters more than raw count.
- Hybrid approaches (compute culling for most particles + hardware queries for key groups).
*/

use powrush_faction_dynamics::{Faction, FactionVisualIdentity, ParticleParams};

/// Result from a hardware occlusion query.
#[derive(Debug, Clone, Copy)]
pub struct OcclusionQueryResult {
    pub samples_passed: u64,
    pub is_occluded: bool,
}

impl OcclusionQueryResult {
    pub fn new(samples_passed: u64) -> Self {
        Self {
            samples_passed,
            is_occluded: samples_passed == 0,
        }
    }
}

/// Helper to decide whether to use hardware queries or compute culling
/// for a given particle effect.
pub fn should_use_hardware_query(
    particle_count: u32,
    screen_coverage: f32, // estimated screen size of the effect
    is_important: bool,
) -> bool {
    // Heuristic: Use hardware queries for large, important effects
    if is_important && screen_coverage > 0.05 {
        return true;
    }
    // For many small particles, prefer compute culling
    if particle_count > 256 {
        return false;
    }
    true
}

pub mod compute { /* existing advanced culling shaders */ }

pub mod wgsl { /* existing shader code */ }
