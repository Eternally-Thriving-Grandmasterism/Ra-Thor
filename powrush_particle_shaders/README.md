# Powrush Particle Shaders

## vkCmdDrawIndirectCount Integration (Host Side)

### Recommended Host-Side Flow

```c
// 1. Create buffers
VkBuffer draw_commands = create_indirect_buffer(max_draws);
VkBuffer draw_count    = create_draw_count_buffer(); // contains u32
VkBuffer visible_indices = create_storage_buffer(...);

// 2. After running DISTANCE_AND_HIZ_TEST + COMPACTION_WITH_DRAW_COUNT
//    (with proper memory barriers)

VkBufferMemoryBarrier barrier = { ... };
vkCmdPipelineBarrier(cmd, ...);

// 3. Record the indirect draw with GPU-provided count
vkCmdDrawIndirectCount(
    cmd,
    draw_commands,           // Buffer with DrawIndirect structures
    0,                       // offset
    draw_count,              // Buffer containing the actual draw count
    0,                       // offset to count
    max_draws,               // maximum possible draws
    sizeof(VkDrawIndirectCommand)
);
```

### Key Points

- The `draw_count` buffer is written atomically by `COMPACTION_WITH_DRAW_COUNT`.
- Use proper pipeline barriers between the compaction compute pass and the draw command.
- This removes the need for the CPU to read back the instance count.

---
*GPU-Driven Rendering*