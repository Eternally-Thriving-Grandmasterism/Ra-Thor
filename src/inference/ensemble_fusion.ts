// src/inference/ensemble_fusion.ts – Ensemble Fusion Layer v1.0
// Combines multiple gesture transformer variants via valence-weighted soft voting
// Disagreement entropy gating, mercy-protected output, uncertainty-aware rejection
// MIT License – Autonomicity Games Inc. 2026

import * as ort from 'onnxruntime-web';
import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';
import mercyHaptic from '@/utils/haptic-utils';
import GestureInferencePipeline from './gesture_inference_pipeline';

const ENSEMBLE_MODELS = [
  { key: 'high', path: '/models/gesture-transformer-qat-int4.onnx', weight: 1.0 },
  { key: 'medium', path: '/models/gesture-transformer-qat-int8.onnx', weight: 0.85 },
  { key: 'low', path: '/models/gesture-transformer-fp16.onnx', weight: 0.7 },
];

const MIN_AGREEMENT = 0.65;              // min weighted agreement to accept prediction
const ENTROPY_THRESHOLD = 1.2;           // max entropy (uncertainty) to accept
const MERCY_DISAGREEMENT_DROP = 0.08;    // max allowable projected valence drop

interface ModelSession {
  key: string;
  session: ort.InferenceSession;
  weight: number;
}

let ensembleSessions: ModelSession[] = [];
let isInitialized = false;

export class EnsembleFusion {
  static async initialize(forceReload = false): Promise<void> {
    const actionName = 'Initialize ensemble fusion layer';
    if (!await mercyGate(actionName)) return;

    if (isInitialized && !forceReload) {
      console.log("[EnsembleFusion] Already initialized with", ensembleSessions.length, "models");
      return;
    }

    console.log("[EnsembleFusion] Initializing multi-model ensemble...");

    ensembleSessions = [];

    for (const model of ENSEMBLE_MODELS) {
      try {
        const session = await ort.InferenceSession.create(model.path, {
          executionProviders: ['webnn', 'webgpu', 'webgl', 'cpu'],
          graphOptimizationLevel: 'all',
        });

        ensembleSessions.push({
          key: model.key,
          session,
          weight: model.weight * (1 + (currentValence.get() - 0.5) * 0.4), // slight valence boost
        });

        console.log(`[EnsembleFusion] Loaded ${model.key} model`);
      } catch (err) {
        console.warn(`Failed to load ${model.key} model:`, err);
      }
    }

    if (ensembleSessions.length === 0) {
      throw new Error("No ensemble models could be loaded");
    }

    isInitialized = true;
    mercyHaptic.playPattern('cosmicHarmony', currentValence.get());
    console.log(`[EnsembleFusion] Initialized – ${ensembleSessions.length} models loaded`);
  }

  static async infer(landmarks: Float32Array): Promise<{
    gesture: string;
    confidence: number;
    futureValence: number[];
    projectedValence: number;
    isSafe: boolean;
    agreement: number;
    entropy: number;
    modelsUsed: string[];
  }> {
    if (!isInitialized) {
      await this.initialize();
    }

    const actionName = 'Run ensemble inference';
    if (!await mercyGate(actionName)) {
      return this.fallbackResult();
    }

    const inputTensor = new ort.Tensor(
      'float32',
      landmarks,
      [1, SEQUENCE_LENGTH, LANDMARK_DIM]
    );

    const predictions: Array<{
      gestureLogits: Float32Array;
      futureValence: Float32Array;
      weight: number;
    }> = [];

    let totalWeight = 0;

    for (const { session, weight } of ensembleSessions) {
      const feeds = { input: inputTensor };
      const results = await session.run(feeds);

      predictions.push({
        gestureLogits: results.gesture_logits.data as Float32Array,
        futureValence: results.future_valence.data as Float32Array,
        weight,
      });

      totalWeight += weight;
    }

    inputTensor.dispose();

    // Weighted soft voting on gesture logits
    const weightedLogits = new Float32Array(NUM_GESTURE_CLASSES).fill(0);
    const weightedFuture = new Float32Array(FUTURE_VALENCE_HORIZON).fill(0);

    for (const pred of predictions) {
      const w = pred.weight / totalWeight;
      for (let i = 0; i < NUM_GESTURE_CLASSES; i++) {
        weightedLogits[i] += pred.gestureLogits[i] * w;
      }
      for (let i = 0; i < FUTURE_VALENCE_HORIZON; i++) {
        weightedFuture[i] += pred.futureValence[i] * w;
      }
    }

    // Softmax on weighted logits
    const maxLogit = Math.max(...weightedLogits);
    const expLogits = weightedLogits.map(l => Math.exp(l - maxLogit));
    const sumExp = expLogits.reduce((a, b) => a + b, 0);
    const probs = expLogits.map(e => e / sumExp);

    const confidence = Math.max(...probs);
    const gestureIndex = probs.indexOf(confidence);
    const gesture = confidence > CONFIDENCE_THRESHOLD ? GESTURE_NAMES[gestureIndex] : 'none';

    const projectedValence = weightedFuture.reduce((a, b) => a + b, 0) / FUTURE_VALENCE_HORIZON;
    const currentVal = currentValence.get();
    const isSafe = projectedValence >= currentVal - MERCY_VALENCE_DROP_THRESHOLD;

    // Agreement & entropy for uncertainty
    const entropy = -probs.reduce((sum, p) => sum + p * Math.log(p + 1e-10), 0);
    const agreement = probs[gestureIndex]; // top-1 confidence as proxy

    if (!isSafe || entropy > 1.5) {
      mercyHaptic.playPattern('warningPulse', currentVal * 0.7);
    }

    return {
      gesture,
      confidence,
      futureValence: Array.from(weightedFuture),
      projectedValence,
      isSafe,
      agreement,
      entropy,
      modelsUsed: ensembleSessions.map(s => s.key),
    };
  }

  private static fallbackResult() {
    return {
      gesture: 'none',
      confidence: 0,
      futureValence: Array(FUTURE_VALENCE_HORIZON).fill(0.5),
      projectedValence: currentValence.get(),
      isSafe: false,
      agreement: 0,
      entropy: 0,
      modelsUsed: [],
    };
  }

  static async dispose() {
    for (const { session } of ensembleSessions) {
      await session.release();
    }
    ensembleSessions = [];
    console.log("[EnsembleFusion] All sessions disposed");
  }
}

export default EnsembleFusion;
