// src/sync/crdt-conflict-strategies-deep-dive.ts – CRDT Conflict Strategies Deep Dive Reference & Mercy Helpers v1
// Detailed Yjs/Automerge/ElectricSQL strategies, valence-weighted overrides, mercy gates
// MIT License – Autonomicity Games Inc. 2026

import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';

const MERCY_THRESHOLD = 0.9999999;

/**
 * CRDT conflict strategies reference – mercy-aligned summary
 */
export const CRDTConflictStrategiesReference = {
  yjs_yata: "Origin pointers + clientID tie-breaker → very strong intention preservation (lists/text)",
  automerge_lists: "Concurrent inserts → ordered by (actorId, seq) Lamport pair",
  automerge_maps: "Last writer wins per key (highest (actorId, seq))",
  automerge_counters: "All concurrent increments summed (commutative)",
  electricsql_default: "Last-writer-wins per row/column via LC timestamp (wallClockMs, clientId, txId)",
  electricsql_custom: "Per-column SQL resolvers (CASE, GREATEST, valence-weighted)",
  mercy_override: "Valence-weighted semantic resolver: higher valence change wins; thriving implication check required",
  multiplanetary_note: "High-latency → queue changes → auto-replay on reconnect; valence filter discards low-thriving deltas"
};

/**
 * Unified valence-weighted semantic resolver (used across engines)
 */
export function valenceWeightedResolver(context: {
  key: string;
  localValue: any;
  localValence: number;
  remoteValue: any;
  remoteValence: number;
}): any {
  const actionName = `Valence-weighted conflict resolution for ${context.key}`;
  if (!mercyGate(actionName, context.key)) {
    return context.localValue; // client preference fallback
  }

  if (context.localValence > context.remoteValence + 0.05) {
    console.log(`[MercyCRDT] Valence wins: local (\( {context.localValence.toFixed(4)}) > remote ( \){context.remoteValence.toFixed(4)})`);
    return context.localValue;
  } else if (context.remoteValence > context.localValence + 0.05) {
    console.log(`[MercyCRDT] Valence wins: remote (\( {context.remoteValence.toFixed(4)}) > local ( \){context.localValence.toFixed(4)})`);
    return context.remoteValue;
  }

  // Fallback to native engine rules (timestamp/clientId/LWW)
  return context.localValue; // client preference
}
