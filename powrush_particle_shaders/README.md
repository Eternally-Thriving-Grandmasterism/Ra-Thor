# Powrush Particle Shaders

## Compute Pipeline Manager & Validation Layers

The manager now includes awareness of Vulkan Validation Layers:

- `ValidationFeatures` struct for configuring debug printf, GPU-assisted validation, etc.
- Guidance on enabling `VK_LAYER_KHRONOS_validation` with `VkValidationFeaturesEXT` at instance creation time.

For full SPIR-V validation and `debugPrintfEXT` support in shaders, enable the validation layer with the appropriate features when creating the Vulkan instance.

---
*Phase 1 Consolidation*