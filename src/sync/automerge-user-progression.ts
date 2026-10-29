// src/sync/automerge-user-progression.ts – Automerge per-user progression document v1
// Durable valence ladder, experience, badges, offline-first multi-device sync
// MIT License – Autonomicity Games Inc. 2026

import * as Automerge from '@automerge/automerge';
import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';

const PROGRESS_DOC_KEY = 'user-progression';

export class AutomergeUserProgression {
  private doc: Automerge.Doc<any>;

  constructor(initialDoc?: Uint8Array) {
    if (initialDoc) {
      this.doc = Automerge.load(initialDoc);
    } else {
      this.doc = Automerge.from({
        valence: currentValence.get(),
        level: 'Newcomer',
        experience: 0,
        badges: [],
        lastActivity: Date.now()
      });
    }
  }

  async updateProgress(deltaValence: number, experienceGain: number, badge?: string) {
    if (!await mercyGate('Update user progression', 'EternalThriving')) return;

    Automerge.change(this.doc, 'Update progression', d => {
      d.valence = Math.min(1.0, d.valence + deltaValence);
      d.experience += experienceGain;
      if (badge && !d.badges.includes(badge)) {
        d.badges.push(badge);
      }
      d.lastActivity = Date.now();
    });

    console.log(`[AutomergeProgress] Updated – valence ${this.getValence().toFixed(4)}, exp ${this.getExperience()}`);
  }

  getValence(): number {
    return Automerge.get(this.doc, ['valence']) ?? 0.5;
  }

  getExperience(): number {
    return Automerge.get(this.doc, ['experience']) ?? 0;
  }

  getBinarySnapshot(): Uint8Array {
    return Automerge.save(this.doc);
  }

  static loadFromBinary(binary: Uint8Array): AutomergeUserProgression {
    return new AutomergeUserProgression(binary);
  }
}

// Singleton instance (or per-user in multi-user setup)
export const userProgression = new AutomergeUserProgression();

// Usage in dashboard / onboarding
// await userProgression.updateProgress(0.02, 50, 'Pinch Pioneer');
// const binary = userProgression.getBinarySnapshot();
// multiplanetarySync.syncState({ progressionBinary: binary });
