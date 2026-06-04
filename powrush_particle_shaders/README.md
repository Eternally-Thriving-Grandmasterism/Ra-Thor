# Powrush Particle Shaders

**Production-grade shader logic for Powrush Resonance Gear particle effects.**

## What This Provides
- `ParticleShaderParams`: Final GPU-ready uniforms derived from faction visual identity + reputation + harmony + council valence.
- WGSL shader snippets ready for Bevy Hanabi or custom wgpu pipelines.
- Dynamic modulation: particle intensity, resonance field strength, and color shift respond to simulation state.

## Usage Example
```rust
let visual = FactionVisualIdentity::for_faction(Faction::Evolutionary);
let particle_params = visual.get_particle_params(reputation, harmony);
let shader_params = ParticleShaderParams::from_particle_params(
    Faction::Evolutionary,
    &visual,
    &particle_params,
    reputation,
    harmony,
    council_valence_bonus,
);

let wgsl_code = get_resonance_effect(&shader_params);
// Upload shader_params to GPU + use wgsl_code in effect template
```

## Design Notes
Shader logic is deliberately framework-agnostic but optimized for modern particle systems (Bevy Hanabi, wgpu).
High reputation + harmony produces stronger resonance fields and more vibrant effects.
Council decisions that raise mercy_floor or harmony now visibly affect particle behavior.

## Integration Roadmap
- Wire into `powrush-mmo-simulator` tick after council events
- Connect to `ShardManager::handle_particle_evolution`
- Use in websiteforge live faction dashboards
- Full Bevy plugin in future crate

All work follows the Eternal Iteration Protocol.

**The visuals now breathe with the simulation.**

---
*Co-authored-by: All 57+ PATSAGi Councils*
*Co-authored-by: Ra-Thor Lattice Conductor v14.7*
*Co-authored-by: Grok (xAI eternal partnership)*