# Powrush Particle Shaders

## vkCmdDrawIndirectCount Support

Added `COMPACTION_WITH_DRAW_COUNT` shader.

This version writes the actual number of draws into a separate `draw_count` buffer,
enabling the use of `vkCmdDrawIndirectCount` on the host side for fully GPU-driven draw submission.

Host-side usage:
```c
vkCmdDrawIndirectCount(
    cmd,
    drawCommandsBuffer,
    0,
    drawCountBuffer,   // written by COMPACTION_WITH_DRAW_COUNT
    0,
    maxDrawCount,
    sizeof(VkDrawIndirectCommand)
);
```

---
*GPU-Driven Rendering*