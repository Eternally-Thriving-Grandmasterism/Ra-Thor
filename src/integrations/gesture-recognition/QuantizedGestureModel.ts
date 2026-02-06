// src/integrations/gesture-recognition/QuantizedGestureModel.ts – Quantized Custom Transformer Loader v6
// QAT-aware preference (trained with fake-quant), layered fallback, mercy-gated
// MIT License – Autonomicity Games Inc. 2026

import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgl';
import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';

const MERCY_THRESHOLD = 0.9999999;
const QAT_4BIT_URL = '/models/gesture-transformer-qat-4bit/model.json';       // QAT 4-bit
const QAT_2BIT_URL = '/models/gesture-transformer-qat-2bit/model.json';       // QAT 2-bit
const TERNARY_QAT_URL = '/models/gesture-transformer-ternary-qat/model.json'; // QAT ternary
const QUANTIZED_8BIT_URL = '/models/gesture-transformer-8bit-int8/model.json';
const FULL_FP16_URL     = '/models/gesture-transformer-full/model.json';

let modelPromise: Promise<tf.LayersModel> | null = null;

export class QuantizedGestureModel {
  static async load(): Promise<tf.LayersModel> {
    const actionName = 'Load QAT-quantized custom transformer model';
    if (!await mercyGate(actionName)) {
      throw new Error("Mercy gate blocked model loading");
    }

    if (modelPromise) return modelPromise;

    const valence = currentValence.get();
    let selectedUrl = FULL_FP16_URL;

    if (valence > 0.97) {
      // Ultra-high valence → prefer QAT ternary (extreme speed + thriving recovery)
      selectedUrl = TERNARY_QAT_URL;
    } else if (valence > 0.94) {
      // Very high valence → QAT 2-bit
      selectedUrl = QAT_2BIT_URL;
    } else if (valence > 0.90) {
      // High valence → QAT 4-bit (best QAT speed/quality)
      selectedUrl = QAT_4BIT_URL;
    } else if (valence > 0.82) {
      // Medium valence → PTQ 8-bit
      selectedUrl = QUANTIZED_8BIT_URL;
    }

    console.log(`[QuantizedGestureModel] Loading QAT-aware model ${selectedUrl} for valence ${valence.toFixed(4)}`);

    try {
      modelPromise = tf.loadLayersModel(selectedUrl);
      const model = await modelPromise;

      // Warm-up inference
      const dummyInput = tf.zeros([1, SEQUENCE_LENGTH, LANDMARK_DIM]);
      const dummyOutput = model.predict(dummyInput) as tf.Tensor[];
      dummyOutput.forEach(t => t.dispose());
      dummyInput.dispose();

      console.log("[QuantizedGestureModel] QAT model loaded & warmed up successfully");
      return model;
    } catch (e) {
      console.error("[QuantizedGestureModel] Load failed", e);
      // Fallback chain
      const fallbackUrls = [QAT_2BIT_URL, QAT_4BIT_URL, QUANTIZED_8BIT_URL, FULL_FP16_URL];
      for (const url of fallbackUrls) {
        try {
          modelPromise = tf.loadLayersModel(url);
          return await modelPromise;
        } catch {}
      }
      throw new Error("All model loading attempts failed");
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
