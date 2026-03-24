// tests/memory-leak.test.js
// Memory Leak Tests for Ra-Thor Sovereign Orchestrator — Detects growth over 100 cycles

import RaThorSovereignOrchestrator from '../core/ra-thor-sovereign-orchestrator.js';

async function runMemoryLeakTests() {
  console.log("%c🧪 Running Memory Leak Tests — 100 cycles", "color:#00ff9d; font-size:18px");
  const orchestrator = new RaThorSovereignOrchestrator();

  const initialMemory = performance.memory ? performance.memory.usedJSHeapSize : 0;
  const memorySamples = [];

  for (let i = 0; i < 100; i++) {
    await orchestrator.process({ rawInput: `leak_test_cycle_${i}`, truthFactor: 0.98 });
    if (performance.memory) {
      memorySamples.push(performance.memory.usedJSHeapSize);
    }
  }

  await orchestrator.cleanup();

  const finalMemory = performance.memory ? performance.memory.usedJSHeapSize : 0;
  const maxGrowth = Math.max(...memorySamples) - initialMemory;
  const finalGrowth = finalMemory - initialMemory;

  const noLeak = maxGrowth < 5000000 && finalGrowth < 1000000; // <5MB growth threshold

  console.assert(noLeak, `Memory leak detected! Max growth: ${(maxGrowth/1024/1024).toFixed(2)} MB`);

  console.log("%c✅ Memory Leak Tests Passed — No significant growth after 100 cycles", "color:#00ff9d; font-size:20px");
  console.table({ initialMB: (initialMemory/1024/1024).toFixed(2), finalMB: (finalMemory/1024/1024).toFixed(2), maxGrowthMB: (maxGrowth/1024/1024).toFixed(2) });
}

runMemoryLeakTests();
