# Powrush Particle Shaders

## Compute Pipeline Manager

Further refined with SPIR-V shader module loading support:

- `load_shader_module(pipeline_type, spirv_bytes)`
- `load_shader_module_from_file(pipeline_type, path)`

The manager can now load shaders directly from SPIR-V bytecode or files.

---
*Phase 1 Consolidation*