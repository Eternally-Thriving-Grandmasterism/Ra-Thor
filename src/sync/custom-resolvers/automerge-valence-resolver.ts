// src/sync/custom-resolvers/automerge-valence-resolver.ts – Automerge-specific valence resolver wrapper v1
// For map key conflict resolution
// MIT License – Autonomicity Games Inc. 2026

import * as Automerge from '@automerge/automerge';
import { unifiedValenceResolver } from './unified-valence-resolver';

export function automergeValenceResolver(
  key: string,
  localChange: Automerge.Change<any>,
  remoteChange: Automerge.Change<any>
): Automerge.Change<any> | null {
  const ctx = {
    key,
    localValue: localChange.value,
    localValence: localChange.valence ?? currentValence.get(),
    localTimestamp: localChange.timestamp,
    remoteValue: remoteChange.value,
    remoteValence: remoteChange.valence ?? currentValence.get(),
    remoteTimestamp: remoteChange.timestamp
  };

  const resolved = unifiedValenceResolver(ctx);
  if (resolved === localChange.value) return localChange;
  if (resolved === remoteChange.value) return remoteChange;

  return null; // fallback to native LWW
}
