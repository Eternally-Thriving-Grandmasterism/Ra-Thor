// src/sync/replicache-triplit-bridge.ts – Replicache + Triplit Hybrid Bridge Layer v1
// Optimistic Replicache mutations mirrored to durable Triplit DB, valence prioritization, rollback on rejection
// MIT License – Autonomicity Games Inc. 2026

import Replicache from 'replicache';
import { Client } from '@triplit/client';
import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';
import mercyHaptic from '@/utils/haptic-utils';

const MERCY_THRESHOLD = 0.9999999;
const VALENCE_MUTATION_PIVOT = 0.9;
const VALENCE_QUERY_PIVOT = 0.9;

interface BridgeConfig {
  replicache: Replicache<any>;
  triplit: Client<any>;
}

let bridge: BridgeConfig | null = null;

export class ReplicacheTriplitBridge {
  static async initialize(replicache: Replicache<any>, triplit: Client<any>) {
    const actionName = 'Initialize Replicache-Triplit hybrid bridge';
    if (!await mercyGate(actionName)) return;

    bridge = { replicache, triplit };

    // 1. Mirror high-valence Replicache mutations to Triplit
    replicache.on('change', async () => {
      const valence = currentValence.get();
      if (valence < VALENCE_MUTATION_PIVOT) return; // low valence → defer to batch

      // Get pending optimistic mutations
      const pending = await replicache.pendingMutations();
      for (const mut of pending) {
        const { name, args } = mut;
        await this.mirrorToTriplit(name, args, valence);
      }
    });

    // 2. Replicache push success → confirm Triplit sync
    replicache.on('push', async (mutations) => {
      const valence = currentValence.get();
      if (valence > VALENCE_MUTATION_PIVOT) {
        mercyHaptic.playPattern('cosmicHarmony', valence);
      }
    });

    // 3. Replicache push rejected → rollback & notify
    replicache.on('push-rejected', async (mutations) => {
      mercyHaptic.playPattern('warningPulse', 0.7);
      console.warn("[ReplicacheTriplitBridge] Server rejected mutations – rollback triggered");
      // Replicache auto-rolls back optimistic changes
    });

    console.log("[ReplicacheTriplitBridge] Initialized – optimistic ↔ durable sync active");
  }

  private static async mirrorToTriplit(mutationName: string, args: any, valence: number) {
    if (!bridge) return;

    const { triplit } = bridge;

    try {
      switch (mutationName) {
        case 'setValence':
          await triplit.insert('valence_logs', {
            id: crypto.randomUUID(),
            user_id: args.userId,
            valence: args.value,
            timestamp: new Date(),
            source: 'optimistic-mutation'
          });
          break;

        case 'setProgressLevel':
          await triplit.upsert('progress_ladders', args.userId, {
            level: args.level,
            description: args.description,
            updated_at: new Date()
          });
          break;

        case 'logGesture':
          await triplit.insert('gesture_logs', {
            id: crypto.randomUUID(),
            type: args.type,
            confidence: args.confidence,
            valence,
            timestamp: new Date()
          });
          break;

        default:
          console.warn(`[ReplicacheTriplitBridge] Unknown mutation: ${mutationName}`);
      }

      if (valence > VALENCE_MUTATION_PIVOT) {
        // High valence → trigger immediate Triplit sync
        await triplit.syncEngine.sync();
      }
    } catch (e) {
      console.error("[ReplicacheTriplitBridge] Mirror to Triplit failed", e);
    }
  }

  static async queryWithValencePriority(queryFn: (client: Client<any>) => Promise<any>) {
    if (!bridge) return null;

    const valence = currentValence.get();
    let result = await queryFn(bridge.triplit);

    if (valence > VALENCE_QUERY_PIVOT) {
      // Prioritize fresh server data
      await bridge.triplit.syncEngine.sync();
      result = await queryFn(bridge.triplit);
    }

    return result;
  }

  static getStatus() {
    return {
      replicacheOnline: bridge?.replicache?.online || false,
      triplitConnected: bridge?.triplit?.syncEngine.isConnected || false,
      pendingReplicache: bridge?.replicache?.pendingMutationsCount || 0,
      lastValenceMutation: currentValence.get()
    };
  }
}

export default ReplicacheTriplitBridge;
