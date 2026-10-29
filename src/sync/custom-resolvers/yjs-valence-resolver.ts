// src/sync/custom-resolvers/yjs-valence-resolver.ts – Yjs-specific valence resolver wrapper v1
// For custom item merge behavior in Y.Map / Y.Array
// MIT License – Autonomicity Games Inc. 2026

import * as Y from 'yjs';
import { unifiedValenceResolver } from './unified-valence-resolver';

export function yjsValenceResolver(
  key: string,
  localItem: Y.Item,
  remoteItem: Y.Item
): Y.Item | null {
  const localVal = localItem.content?.getContent()?.[0];
  const remoteVal = remoteItem.content?.getContent()?.[0];

  const ctx = {
    key,
    localValue: localVal,
    localValence: currentValence.get(),
    localTimestamp: localItem.id.clock,
    remoteValue: remoteVal,
    remoteValence: currentValence.get(), // proxy – real impl carries valence metadata
    remoteTimestamp: remoteItem.id.clock
  };

  const resolvedValue = unifiedValenceResolver(ctx);
  if (resolvedValue === localVal) return localItem;
  if (resolvedValue === remoteVal) return remoteItem;

  // Let native YATA decide
  return null;
}
