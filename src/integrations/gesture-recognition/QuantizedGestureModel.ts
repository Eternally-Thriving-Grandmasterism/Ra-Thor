// src/integrations/gesture-recognition/QuantizedGestureModel.ts – Quantized Custom Transformer Loader v1
// Loads 4-bit AWQ quantized model first, fallback to 8-bit/FP16, mercy-gated
// MIT License – Autonomicity Games Inc. 2026

import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgl';
import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';

const MERCY_THRESHOLD = 0.9999999;
const QUANTIZED_4BIT_URL = '/models/gesture-transformer-4bit-awq/model.json';
const QUANTIZED_8BIT_URL = '/models/gesture-transformer-8bit-int8/model.json';
const FULL_FP16_URL = '/models/gesture-transformer-full/model.json';

let modelPromise: Promise<tf.LayersModel> | null = null;

export class QuantizedGestureModel {
  static async load(): Promise<tf.LayersModel> {
    const actionName = 'Load quantized custom transformer model';
    if (!await mercyGate(actionName)) {
      throw new Error("Mercy gate blocked model loading");
    }

    if (modelPromise) return modelPromise;

    const valence = currentValence.get();
    let selectedUrl = FULL_FP16_URL;

    if (valence > 0.92) {
      // High valence → prefer 4-bit AWQ (fastest + thriving-aligned)
      selectedUrl = QUANTIZED_4BIT_URL;
    } else if (valence > 0.85) {
      // Medium valence → 8-bit int8 (balanced speed/safety)
      selectedUrl = QUANTIZED_8BIT_URL;
    } else {
      // Low valence → full FP16 (maximum accuracy for survival mode)
      selectedUrl = FULL_FP16_URL;
    }

    console.log(`[QuantizedGestureModel] Loading model (${selectedUrl}) for valence ${valence.toFixed(4)}`);

    try {
      modelPromise = tf.loadLayersModel(selectedUrl);
      const model = await modelPromise;

      // Warm up backend
      const dummyInput = tf.zeros([1, SEQUENCE_LENGTH, LANDMARK_DIM]);
      await model.predict(dummyInput);
      dummyInput.dispose();

      console.log("[QuantizedGestureModel] Model loaded & warmed up successfully");
      return model;
    } catch (e) {
      console.error("[QuantizedGestureModel] Load failed", e);
      // Fallback to full model on error
      modelPromise = tf.loadLayersModel(FULL_FP16_URL);
      return await modelPromise;
    }
  }

  static async dispose() {
    if (modelPromise) {
      const model = await modelPromise;
      model.dispose();
      modelPromise = null;
      console.log("[QuantizedGestureModel] Model disposed");
    }
  }
}

export default QuantizedGestureModel;
