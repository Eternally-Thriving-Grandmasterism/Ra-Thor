// src/sync/electricsql-progress-sync.ts – ElectricSQL progress & valence sync v1
// Real-time relational updates, offline queue, mercy-gated logging
// MIT License – Autonomicity Games Inc. 2026

import { electricInitializer } from './electricsql-initializer';
import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';

export async function syncProgressToElectric(deltaValence: number, deltaExperience: number, eventType: string) {
  const actionName = `Sync progress event: ${eventType}`;
  if (!await mercyGate(actionName)) return;

  const electric = electricInitializer.getElectricClient();
  if (!electric) return;

  const userId = 'current-user'; // replace with real user ID in multi-user

  // Upsert user record
  await electric.db.users.upsert({
    id: userId,
    level: 'Ultramaster', // dynamic in real impl
    valence: currentValence.get(),
    experience: (await electric.db.users.findFirst({ where: { id: userId } }))?.experience ?? 0 + deltaExperience,
    lastActivity: Date.now(),
    createdAt: Date.now()
  });

  // Log event
  await electric.db.progress_logs.create({
    data: {
      id: crypto.randomUUID(),
      userId,
      eventType,
      deltaValence,
      deltaExperience,
      timestamp: Date.now()
    }
  });

  // Trigger sync (ElectricSQL handles batching & offline queue)
  await electric.sync();

  console.log(`[ElectricProgressSync] Synced ${eventType} – Δvalence ${deltaValence.toFixed(4)}, Δexp ${deltaExperience}`);
}

// Example usage after daily pulse
// await syncProgressToElectric(0.02, 50, 'Daily Mercy Pulse');
