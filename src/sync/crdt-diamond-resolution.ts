// src/sync/crdt-diamond-resolution.ts – Mercy Diamond CRDT Conflict Resolution Reference & Helpers v1
// Detailed Yjs/Automerge diamond merge semantics, valence-weighted custom resolver
// MIT License – Autonomicity Games Inc. 2026

import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';

const MERCY_THRESHOLD = 0.9999999;

/**
 * Diamond conflict resolution reference – Yjs (primary) vs Automerge (subdocs)
 */
export const DiamondResolutionSemantics = {
  yjs: {
    lists: "Concurrent inserts after same origin → ordered by clientID tie-breaker (lower ID first)",
    maps: "Last writer wins per key (Lamport timestamp + clientID)",
    text: "Span-based last-writer-wins + tombstones",
    subdocs: "Independent Y.Doc instances – no cross-history conflicts",
    overall: "Intention-preserving, deterministic, no user-visible conflicts"
  },
  automerge: {
    lists: "Concurrent inserts at same position → Lamport timestamp + actor ID order",
    maps: "Last writer wins per key (highest Lamport)",
    counters: "All concurrent increments summed (commutative)",
    subdocs: "Independent Automerge docs – binary blobs in parent, no cross-conflicts",
    overall: "Deterministic, intention-preserving, strong for nested independent state"
  },
  mercy_override: "Valence-weighted resolver for critical semantic conflicts: higher valence change wins"
};

/**
 * Valence-weighted conflict resolver for concurrent map updates
 * Higher valence change preferred in rare concurrent updates
 */
export function valenceWeightedResolver(
  key: string,
  localChange: { value: any; valence: number; timestamp: number },
  remoteChange: { value: any; valence: number; timestamp: number }
): any {
  const actionName = `Resolve diamond conflict for key: ${key}`;
  if (!mercyGate(actionName, key)) {
    // Fallback to timestamp
    return localChange.timestamp > remoteChange.timestamp ? localChange.value : remoteChange.value;
  }

  if (localChange.valence > remoteChange.valence + 0.05) {
    console.log(`[MercyCRDT] Diamond conflict resolved: local wins (valence ${localChange.valence.toFixed(4)})`);
    return localChange.value;
  } else if (remoteChange.valence > localChange.valence + 0.05) {
    console.log(`[MercyCRDT] Diamond conflict resolved: remote wins (valence ${remoteChange.valence.toFixed(4)})`);
    return remoteChange.value;
  }

  // Tie → Lamport timestamp
  return localChange.timestamp > remoteChange.timestamp ? localChange.value : remoteChange.value;
}

// Usage example in sync handler
// const resolvedValue = valenceWeightedResolver('probe-001-resources', localChange, remoteChange);
