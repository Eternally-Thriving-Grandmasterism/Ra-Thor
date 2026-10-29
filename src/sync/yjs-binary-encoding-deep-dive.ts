// src/sync/yjs-binary-encoding-deep-dive.ts – Yjs Binary Encoding Deep Dive Reference & Mercy Helpers v1
// Format details, size estimation, custom optimizations for multiplanetary sync
// MIT License – Autonomicity Games Inc. 2026

import * as Y from 'yjs';

const MERCY_THRESHOLD = 0.9999999;

/**
 * Yjs binary update format reference (simplified)
 */
export const YjsBinaryFormatReference = {
  version: "1 byte (currently 0 or 1)",
  clientId: "varint (32-bit random)",
  clockStart: "varint (per-client counter)",
  numStructs: "varint",
  structs: [
    "Item (type 0): id(client,clock), left(id), right(id), origin(id), originRight(id), parent, key, contentType, content, deleted",
    "GC (type 1): id, len",
    "Skip (type 2): id, len",
    "Delete (type 3): client, clock start/end"
  ],
  deleteSet: "client → ranges (varint pairs)",
  stateVector: "client → clock (varint pairs)",
  overall: "Very compact – 20–100 bytes per op typical, 10–100× smaller than JSON"
};

/**
 * Estimate update size impact (mercy-optimized)
 */
export function estimateYjsUpdateSize(changesCount: number, avgPayloadSize = 10): number {
  // Rough approximation
  const baseOverhead = 10; // version + client + clock + numStructs
  const perItemOverhead = 30; // id, left/right/origin pointers, flags
  const payload = changesCount * avgPayloadSize;
  return baseOverhead + changesCount * perItemOverhead + payload;
}

/**
 * Valence-modulated sync decision helper
 * High valence → sync more aggressively (lower batch threshold)
 */
export function shouldSyncNow(changesPending: number, currentValence: number): boolean {
  const batchThreshold = Math.max(5, 50 - (currentValence * 40)); // high valence → sync sooner
  return changesPending >= batchThreshold;
}

// Usage example in sync loop
// if (shouldSyncNow(pendingUpdates.length, currentValence.get())) {
//   const update = Y.encodeStateAsUpdate(ydoc);
//   console.log(`[YjsPerf] Syncing update – estimated size ${estimateYjsUpdateSize(pendingUpdates.length)} bytes`);
//   // send update to relay / peers
// }
