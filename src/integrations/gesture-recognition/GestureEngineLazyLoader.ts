// src/integrations/gesture-recognition/GestureEngineLazyLoader.ts – tfjs + BlazePose Lazy Loader v1.1
// Deferred import, quantized model preference, mercy-gated activation
// MIT License – Autonomicity Games Inc. 2026

import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';
import mercyHaptic from '@/utils/haptic-utils';

const MERCY_THRESHOLD = 0.9999999;
const LAZY_ACTIVATION_VALENCE = 0.85;
const WASM_CDN_PREFIX = 'https://cdn.jsdelivr.net/npm/@mediapipe/holistic/';
const QUANTIZED_MODEL_URL = '/models/blazepose-quantized/model.json'; // 4-bit or 8-bit version
const FULL_MODEL_URL = '/models/blazepose-full/model.json'; // fallback FP16

let isLoaded = false;
let holisticPromise: Promise<any> | null = null;
let tfPromise: Promise<typeof import('@tensorflow/tfjs')> | null = null;

export class GestureEngineLazyLoader {
  static async activate(onReady?: () => void) {
    const actionName = 'Lazy-load optimized tfjs & BlazePose engine';
    if (!await mercyGate(actionName)) return;

    if (isLoaded) {
      console.log("[GestureLazyLoader] Already activated");
      onReady?.();
      return;
    }

    console.log("[GestureLazyLoader] Activating optimized tfjs + BlazePose (first load)...");

    try {
      // 1. Load tfjs core + webgl backend
      tfPromise = import('@tensorflow/tfjs').then(async tf => {
        await tf.setBackend('webgl');
        await tf.ready();
        console.log("[GestureLazyLoader] tfjs backend ready:", tf.getBackend());
        return tf;
      });

      // 2. Load quantized BlazePose Holistic first (valence preference)
      const useQuantized = currentValence.get() > 0.9; // high valence → prefer quantized for speed
      const modelUrl = useQuantized ? QUANTIZED_MODEL_URL : FULL_MODEL_URL;

      holisticPromise = import('@mediapipe/holistic').then(async ({ Holistic }) => {
        const holistic = new Holistic({
          locateFile: (file) => `\( {WASM_CDN_PREFIX} \){file}`
        });

        holistic.setOptions({
          modelComplexity: 1,
          smoothLandmarks: true,
          minDetectionConfidence: 0.7,
          minTrackingConfidence: 0.7
        });

        // Custom model loading (quantized or full)
        await holistic.initializeCustomModel(modelUrl); // hypothetical API – real impl patches load
        console.log(`[GestureLazyLoader] BlazePose loaded (${useQuantized ? 'quantized' : 'full'})`);

        return holistic;
      });

      const [tf, holistic] = await Promise.all([tfPromise, holisticPromise]);

      isLoaded = true;
      mercyHaptic.playPattern('cosmicHarmony', currentValence.get());
      console.log("[GestureLazyLoader] Activation complete – ready for inference");

      onReady?.();
    } catch (e) {
      console.error("[GestureLazyLoader] Activation failed", e);
      mercyHaptic.playPattern('warningPulse', 0.7);
    }
  }

  static getHolistic(): Promise<any> | null {
    return holisticPromise;
  }

  static isActive(): boolean {
    return isLoaded;
  }

  static tryAutoActivate() {
    if (currentValence.get() > LAZY_ACTIVATION_VALENCE && !isLoaded) {
      console.log("[GestureLazyLoader] Auto-activation triggered by high valence");
      this.activate();
    }
  }
}

// Auto-check on valence change
currentValence.subscribe(() => {
  GestureEngineLazyLoader.tryAutoActivate();
});

export default GestureEngineLazyLoader;
