// src/sync/automerge-lamport-deep-dive.ts – Automerge Lamport Timestamps Deep Dive Reference & Mercy Helpers v1
// Detailed causal ordering, conflict resolution, valence-weighted tie-breaker
// MIT License – Autonomicity Games Inc. 2026

import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';

const MERCY_THRESHOLD = 0.9999999;

/**
 * Automerge Lamport timestamp reference – core conflict resolution clock
 */
export const AutomergeLamportReference = {
  timestampStructure: "(actorId: 32-bit random, seq: per-actor monotonic counter)",
  causalOrdering: "If change A happened-before B on same actor → A always before B",
  concurrentTieBreaker: "Concurrent changes → ordered by actorId (lower ID first)",
  mapKeyConflict: "Highest Lamport timestamp wins per key",
  listInsertConflict: "Concurrent inserts after same position → ordered by Lamport + actorId",
  counter: "Commutative increments (Lamport only for causal order)",
  mercy_override: "Valence-weighted tie-breaker for semantic conflicts: higher valence change wins"
};

/**
 * Valence-weighted tie-breaker for concurrent map key updates
 */
export function valenceLamportTieBreaker(
  localChange: { value: any; actorId: number; seq: number; valence: number },
  remoteChange: { value: any; actorId: number; seq: number; valence: number }
): any {
  const actionName = `Automerge Lamport tie-breaker for concurrent change`;
  if (!mercyGate(actionName)) {
    // Fallback to actorId + seq (native Automerge rule)
    if (localChange.actorId !== remoteChange.actorId) {
      return localChange.actorId < remoteChange.actorId ? localChange.value : remoteChange.value;
    }
    return localChange.seq > remoteChange.seq ? localChange.value : remoteChange.value;
  }

  if (localChange.valence > remoteChange.valence + 0.05) {
    console.log(`[MercyAutomergeLamport] Tie-breaker: local wins (valence ${localChange.valence.toFixed(4)})`);
    return localChange.value;
  } else if (remoteChange.valence > localChange.valence + 0.05) {
    console.log(`[MercyAutomergeLamport] Tie-breaker: remote wins (valence ${remoteChange.valence.toFixed(4)})`);
    return remoteChange.value;
  }

  // Fallback to native Automerge Lamport rule
  if (localChange.actorId !== remoteChange.actorId) {
    return localChange.actorId < remoteChange.actorId ? localChange.value : remoteChange.value;
  }
  return localChange.seq > remoteChange.seq ? localChange.value : remoteChange.value;
}

// Usage example in sync handler
// const resolvedValue = valenceLamportTieBreaker(localChange, remoteChange);
