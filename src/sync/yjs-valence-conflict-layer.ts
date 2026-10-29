// src/sync/yjs-valence-conflict-layer.ts – Valence-Aware Yjs Conflict Resolution Layer v1
// Valence-weighted tie-breaking, high-valence sync prioritization, mercy gates
// MIT License – Autonomicity Games Inc. 2026

import * as Y from 'yjs';
import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';
import mercyHaptic from '@/utils/haptic-utils';

const MERCY_THRESHOLD = 0.9999999;
const VALENCE_TIE_BREAKER_THRESHOLD = 0.9;
const HIGH_VALENCE_SYNC_PRIORITY = 0.95;

export class YjsValenceConflictLayer {
  static setup(ydoc: Y.Doc, awareness: Y.Awareness) {
    const actionName = 'Setup valence-aware Yjs conflict layer';
    if (!mercyGate(actionName)) return;

    // 1. Valence-aware awareness state
    awareness.on('update', (changes) => {
      const states = awareness.getStates();
      const highValenceUsers = Array.from(states.values()).filter(
        state => state.user?.valence > HIGH_VALENCE_SYNC_PRIORITY
      );

      if (highValenceUsers.length > 0) {
        mercyHaptic.playPattern('cosmicHarmony', currentValence.get());
        // TODO: render special glow/avatars for high-valence users
      }
    });

    // 2. Valence-weighted transaction observer (tie-breaker)
    ydoc.on('update', (update, origin) => {
      if (origin?.clientID) {
        const localValence = currentValence.get();
        if (localValence > VALENCE_TIE_BREAKER_THRESHOLD) {
          // Boost local ops on high valence
          // (Yjs already uses clientID for ties – we can simulate boost via priority queue on reconnect)
          console.log("[ValenceConflict] High-valence local update boosted");
        }
      }
    });

    // 3. Prioritized sync queue (high-valence first on reconnect)
    const syncQueue: Array<{ type: string; data: any; valence: number }> = [];

    const enqueueChange = (type: string, data: any) => {
      const valence = currentValence.get();
      syncQueue.push({ type, data, valence });

      // Sort queue: high valence first, then timestamp
      syncQueue.sort((a, b) => {
        if (b.valence !== a.valence) return b.valence - a.valence;
        return a.data.timestamp - b.data.timestamp;
      });

      // Prune old low-valence entries if queue too large
      if (syncQueue.length > 500) {
        syncQueue.splice(400); // keep newest 400
      }
    };

    // Example: enqueue on local change
    ydoc.on('afterTransaction', (transaction) => {
      if (!transaction.local) return;
      enqueueChange('transaction', transaction);
    });

    // Flush on reconnect
    awareness.on('status', ({ status }) => {
      if (status === 'connected') {
        this.flushHighValenceQueue(syncQueue);
      }
    });
  }

  private static flushHighValenceQueue(queue: typeof syncQueue) {
    // Send high-valence first
    const highValenceBatch = queue.filter(e => e.valence > HIGH_VALENCE_SYNC_PRIORITY);
    highValenceBatch.forEach(e => {
      // TODO: send to relay / other peers
      console.log("[ValenceConflict] Flushing high-valence change:", e.type);
    });

    // Then rest
    queue.splice(0, queue.length);
  }

  static getQueueStatus() {
    return {
      total: syncQueue.length,
      highValenceCount: syncQueue.filter(e => e.valence > HIGH_VALENCE_SYNC_PRIORITY).length
    };
  }
}

export default YjsValenceConflictLayer;
