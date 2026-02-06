// src/integrations/gesture-recognition/MediaPipeLazyLoader.ts – MediaPipe Holistic WASM Lazy Loader v1
// Deferred WASM download, instantiation, model init — only on explicit activation
// MIT License – Autonomicity Games Inc. 2026

import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';
import mercyHaptic from '@/utils/haptic-utils';

const MERCY_THRESHOLD = 0.9999999;
const LAZY_ACTIVATION_VALENCE = 0.85; // auto-activate above this if intent detected
const WASM_CDN_PREFIX = 'https://cdn.jsdelivr.net/npm/@mediapipe/holistic/';

let isLoaded = false;
let holisticPromise: Promise<any> | null = null;

export class MediaPipeLazyLoader {
  static async activate(onReady?: (holistic: any) => void) {
    const actionName = 'Lazy-load MediaPipe Holistic WASM';
    if (!await mercyGate(actionName)) return;

    if (isLoaded) {
      console.log("[MediaPipeLazyLoader] Already activated");
      if (onReady && holisticPromise) {
        const holistic = await holisticPromise;
        onReady(holistic);
      }
      return;
    }

    console.log("[MediaPipeLazyLoader] Activating MediaPipe Holistic WASM (first load)...");

    try {
      // 1. Dynamic import of @mediapipe/holistic (triggers WASM fetch)
      const { Holistic } = await import('@mediapipe/holistic');

      const holistic = new Holistic({
        locateFile: (file) => `\( {WASM_CDN_PREFIX} \){file}`
      });

      holistic.setOptions({
        modelComplexity: 1,
        smoothLandmarks: true,
        minDetectionConfidence: 0.7,
        minTrackingConfidence: 0.7
      });

      await holistic.initialize();

      isLoaded = true;
      holisticPromise = Promise.resolve(holistic);

      mercyHaptic.playPattern('cosmicHarmony', currentValence.get());
      console.log("[MediaPipeLazyLoader] WASM & Holistic initialized – ready for inference");

      onReady?.(holistic);
    } catch (e) {
      console.error("[MediaPipeLazyLoader] WASM activation failed", e);
      mercyHaptic.playPattern('warningPulse', 0.7);
    }
  }

  static getHolistic(): Promise<any> | null {
    return holisticPromise;
  }

  static isActive(): boolean {
    return isLoaded;
  }

  /**
   * Auto-activation on high valence or user intent (e.g. summon orb click)
   */
  static tryAutoActivate() {
    if (currentValence.get() > LAZY_ACTIVATION_VALENCE && !isLoaded) {
      console.log("[MediaPipeLazyLoader] Auto-activation triggered by high valence");
      this.activate();
    }
  }
}

// Auto-check on valence change (optional background warm-up)
currentValence.subscribe(() => {
  MediaPipeLazyLoader.tryAutoActivate();
});

// Export for use in GestureOverlay / MR components
export default MediaPipeLazyLoader;
