// src/integrations/gesture-recognition/TfjsLazyLoader.ts – TensorFlow.js Lazy Loader v1
// Deferred import of tfjs-core + webgl backend, only on explicit activation
// MIT License – Autonomicity Games Inc. 2026

import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';
import mercyHaptic from '@/utils/haptic-utils';

const MERCY_THRESHOLD = 0.9999999;
const LAZY_ACTIVATION_VALENCE = 0.85; // auto-activate above this if intent detected

let isTfjsLoaded = false;
let tfPromise: Promise<typeof import('@tensorflow/tfjs')> | null = null;

export class TfjsLazyLoader {
  static async activate(onReady?: (tf: typeof import('@tensorflow/tfjs')) => void) {
    const actionName = 'Lazy-load TensorFlow.js core & WebGL backend';
    if (!await mercyGate(actionName)) return;

    if (isTfjsLoaded) {
      console.log("[TfjsLazyLoader] Already activated");
      if (onReady && tfPromise) {
        const tf = await tfPromise;
        onReady(tf);
      }
      return;
    }

    console.log("[TfjsLazyLoader] Activating TensorFlow.js (first load)...");

    try {
      const tf = await import('@tensorflow/tfjs');
      await tf.setBackend('webgl');
      await tf.ready();

      isTfjsLoaded = true;
      tfPromise = Promise.resolve(tf);

      mercyHaptic.playPattern('cosmicHarmony', currentValence.get());
      console.log("[TfjsLazyLoader] tfjs backend ready:", tf.getBackend());

      onReady?.(tf);
    } catch (e) {
      console.error("[TfjsLazyLoader] tfjs activation failed", e);
      mercyHaptic.playPattern('warningPulse', 0.7);
    }
  }

  static getTf(): Promise<typeof import('@tensorflow/tfjs')> | null {
    return tfPromise;
  }

  static isActive(): boolean {
    return isTfjsLoaded;
  }

  /**
   * Auto-activation on high valence or user intent (e.g. summon orb click)
   */
  static tryAutoActivate() {
    if (currentValence.get() > LAZY_ACTIVATION_VALENCE && !isTfjsLoaded) {
      console.log("[TfjsLazyLoader] Auto-activation triggered by high valence");
      this.activate();
    }
  }
}

// Auto-check on valence change (optional background warm-up)
currentValence.subscribe(() => {
  TfjsLazyLoader.tryAutoActivate();
});

// Export for use in GestureOverlay / MR components
export default TfjsLazyLoader;
