// src/sync/automerge-lamport-tie-breaker.ts – Automerge Lamport Tie-Breaker Deep Dive & Mercy Helpers v1
// Detailed tie-breaker rules, valence-weighted semantic override, mercy gates
// MIT License – Autonomicity Games Inc. 2026

import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';

const MERCY_THRESHOLD = 0.9999999;

/**
 * Automerge Lamport tie-breaker reference – core conflict resolution rules
 */
export const LamportTieBreakerSemantics = {
  concurrentInsert: "Same origin → lower actorId first; same actorId → higher seq first",
  concurrentMapUpdate: "Highest (actorId, seq) wins per key",
  collisionSameActorId: "Changes remain distinguishable by seq; tie-breaker falls to seq",
  worstCase: "Arbitrary but consistent order – no data loss, no resurrection",
  mercy_override: "Valence-weighted semantic tie-breaker: higher valence change wins"
};

/**
 * Valence-weighted tie-breaker for concurrent changes
 */
export function valenceLamportTieBreaker<T>(
  local: { actorId: number; seq: number; valence: number; value: T },
  remote: { actorId: number; seq: number; valence: number; value: T }
): T {
  const actionName = `Lamport tie-breaker for concurrent change`;
  if (!mercyGate(actionName)) {
    // Native fallback
    if (local.actorId !== remote.actorId) {
      return local.actorId < remote.actorId ? local.value : remote.value;
    }
    return local.seq > remote.seq ? local.value : remote.value;
  }

  if (local.valence > remote.valence + 0.05) {
    console.log(`[MercyLamport] Tie-breaker: local wins (valence ${local.valence.toFixed(4)})`);
    return local.value;
  } else if (remote.valence > local.valence + 0.05) {
    console.log(`[MercyLamport] Tie-breaker: remote wins (valence ${remote.valence.toFixed(4)})`);
    return remote.value;
  }

  // Native Automerge fallback
  if (local.actorId !== remote.actorId) {
    return local.actorId < remote.actorId ? local.value : remote.value;
  }
  return local.seq > remote.seq ? local.value : remote.value;
}

// Usage example in sync handler
// const resolvedValue = valenceLamportTieBreaker(localChange, remoteChange);
