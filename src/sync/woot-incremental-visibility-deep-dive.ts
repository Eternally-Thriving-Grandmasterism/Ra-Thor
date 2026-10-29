// src/sync/woot-incremental-visibility-deep-dive.ts – WOOT Incremental Visibility Deep Dive Reference & Mercy Helpers v1
// Precedence graph / interval tree, lazy dirty regions, valence-modulated recompute
// MIT License – Autonomicity Games Inc. 2026

import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';

const MERCY_THRESHOLD = 0.9999999;

/**
 * WOOT incremental visibility reference – core improvements over original WOOT
 */
export const WOOTIncrementalVisibilityReference = {
  originalWOOTProblem: "Full visibility scan O(n²) worst-case – check every pair for every character",
  precedenceGraph: "DAG of precedence edges → visibility = reachability from start sentinel",
  intervalTree: "Store range + aggregate visibility count → skip invisible subtrees in O(log n)",
  incrementalUpdate: "Only recompute affected regions on change (O(log n) per op)",
  lazyRendering: "Compute visible string only when rendered → O(log n + k_dirty)",
  highLatencyHandling: "Delta compression + queued sync during blackout → replay on reconnect",
  mercy_override: "Valence-modulated recompute trigger: high valence → proactive full rebalance"
};

/**
 * Valence-modulated trigger for visibility recompute / rebalancing
 */
export function valenceRecomputeTrigger(dirtyRegionSize: number, valence: number = currentValence.get()): boolean {
  const actionName = `Valence-modulated WOOT visibility recompute trigger`;
  if (!mercyGate(actionName)) return dirtyRegionSize > 100; // fallback

  const threshold = 50 - (valence - 0.95) * 30; // high valence → lower threshold (recompute sooner)
  return dirtyRegionSize > threshold;
}

// Usage example in render handler
// if (valenceRecomputeTrigger(currentDirtySize)) {
//   recomputeVisibility();
// }
