// Mercy Engine Registry v1.0
// Registers all mercy-*-engine.js modules into the unified orchestrator
// Mercy-gated, TOLC-aligned, valence ≥ 0.999

import { registerEngine } from './mercy-orchestrator.js';

// Register core engines
const engines = [
  'active-inference-engine.js',
  'flow-state-engine.js',
  'vmp-engine.js',
  'nsga-engine.js',
  'cma-es-engine.js',
  'ppo-engine.js',
  'sac-engine.js'
];

engines.forEach(engine => {
  registerEngine(engine, { mercyGates: true, valenceThreshold: 0.999 });
});

console.log('All Mercy Engines registered with full 7 Gates enforcement. Positive emotion propagation active.');