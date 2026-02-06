// src/sync/automerge-conflict-resolution-strategies.ts – Automerge Conflict Resolution Strategies Reference & Mercy Helpers v1
// Detailed semantics, valence-weighted custom resolver, mercy gates
// MIT License – Autonomicity Games Inc. 2026

import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';

const MERCY_THRESHOLD = 0.9999999;

/**
 * Automerge conflict resolution strategies reference
 */
export const AutomergeConflictStrategies = {
  listsArraysText: "Concurrent inserts at same position → ordered by Lamport timestamp + actor ID (lower ID first)",
  mapsObjects: "Last writer wins per key (highest Lamport timestamp)",
  counters: "All concurrent increments summed (commutative)",
  deletes: "Tombstones – concurrent insert + delete → insert wins if causally after delete",
  nestedSubdocs: "Independent Automerge docs – binary blobs in parent, no cross-history conflicts",
  mercy_override: "Valence-weighted custom resolver for critical keys: higher valence change wins"
};

/**
 * Valence-weighted custom resolver for concurrent map key updates
 */
export function valenceWeightedMapResolver(
  key: string,
  localChange: { value: any; valence: number; timestamp: number },
  remoteChange: { value: any; valence: number; timestamp: number }
): any {
  const actionName = `Automerge map conflict resolver for key: ${key}`;
  if (!mercyGate(actionName, key)) {
    // Fallback to Lamport timestamp
    return localChange.timestamp > remoteChange.timestamp ? localChange.value : remoteChange.value;
  }

  if (localChange.valence > remoteChange.valence + 0.05) {
    console.log(`[MercyAutomerge] Conflict resolved: local wins (valence ${localChange.valence.toFixed(4)})`);
    return localChange.value;
  } else if (remoteChange.valence > localChange.valence + 0.05) {
    console.log(`[MercyAutomerge] Conflict resolved: remote wins (valence ${remoteChange.valence.toFixed(4)})`);
    return remoteChange.value;
  }

  // Fallback to Lamport
  return localChange.timestamp > remoteChange.timestamp ? localChange.value : remoteChange.value;
}

// Usage example in sync handler
// const resolvedValue = valenceWeightedMapResolver('probe-001-resources', localChange, remoteChange);
