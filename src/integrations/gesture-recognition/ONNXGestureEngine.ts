// src/integrations/gesture-recognition/ONNXGestureEngine.ts – ONNX Runtime Web Engine v2.4
// WebNN native preference → WebGPU → WebGL fallback, QAT-quantized models, mercy-gated, valence-modulated
// MIT License – Autonomicity Games Inc. 2026

import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';
import mercyHaptic from '@/utils/haptic-utils';
import * as ort from 'onnxruntime-web';

const MERCY_THRESHOLD = 0.9999999;
const VALENCE_HIGH_NPU_PIVOT = 0.94;     // Prefer INT4 + WebNN on high valence
const VALENCE_SAFE_PIVOT = 0.88;         // INT8 + WebNN/WebGPU
const ONNX_TARGET_QAT_INT4_URL = '/models/gesture-transformer-qat-int4/model.onnx';
const ONNX_TARGET_QAT_INT8_URL = '/models/gesture-transformer-qat-int8/model.onnx';
const ONNX_DRAFT_QAT_INT8_URL = '/models/gesture-draft-qat-int8/model.onnx';
const ONNX_TARGET_FULL_URL = '/models/gesture-transformer-onnx/model.onnx';

let sessionTarget: ort.InferenceSession | null = null;
let sessionDraft: ort.InferenceSession | null = null;
let isONNXReady = false;
let currentProvider = 'none';

export class ONNXGestureEngine {
  static async activate() {
    const actionName = 'Activate ONNX Runtime Web with WebNN preference';
    if (!await mercyGate(actionName)) return;

    if (isONNXReady) {
      console.log("[ONNXGestureEngine] Already activated – provider:", currentProvider);
      return;
    }

    console.log("[ONNXGestureEngine] Activating ONNX Runtime Web...");

    try {
      // 1. Configure ORT defaults
      ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/ort-wasm.wasm';
      ort.env.wasm.numThreads = Math.min(navigator.hardwareConcurrency || 4, 6); // cap for mobile
      ort.env.wasm.simd = true;

      const valence = currentValence.get();
      let targetUrl = ONNX_TARGET_FULL_URL;

      // 2. Valence-aware model selection
      if (valence > VALENCE_HIGH_NPU_PIVOT) {
        targetUrl = ONNX_TARGET_QAT_INT4_URL;
      } else if (valence > VALENCE_SAFE_PIVOT) {
        targetUrl = ONNX_TARGET_QAT_INT8_URL;
      }

      // 3. Build provider preference chain
      const providers: string[] = [];

      // WebNN first if available
      const hasWebNN = 'ml' in navigator && 'createContext' in (navigator as any).ml;
      if (hasWebNN && valence > VALENCE_SAFE_PIVOT) {
        providers.push('webnn');
      }

      // WebGPU next
      let hasWebGPU = false;
      try {
        hasWebGPU = 'gpu' in navigator && !!(await (navigator as any).gpu?.requestAdapter?.());
      } catch {}
      if (hasWebGPU) {
        providers.push('webgpu');
      }

      // WebGL fallback
      providers.push('webgl');

      console.log("[ONNXGestureEngine] Trying providers in order:", providers);

      // 4. Try each provider sequentially
      for (const provider of providers) {
        try {
          sessionTarget = await ort.InferenceSession.create(targetUrl, {
            executionProviders: [provider],
            graphOptimizationLevel: 'all',
            enableCpuMemArena: false,     // reduce memory spikes on mobile
            enableMemArena: false
          });

          sessionDraft = await ort.InferenceSession.create(ONNX_DRAFT_QAT_INT8_URL, {
            executionProviders: [provider],
            graphOptimizationLevel: 'all',
            enableCpuMemArena: false,
            enableMemArena: false
          });

          currentProvider = provider;
          break;
        } catch (providerErr) {
          console.warn(`Provider ${provider} failed`, providerErr);
        }
      }

      if (!sessionTarget || !sessionDraft) {
        throw new Error("No suitable execution provider found");
      }

      // 5. Warm-up inference (critical for first-token latency)
      const dummyInput = new ort.Tensor('float32', new Float32Array(SEQUENCE_LENGTH * LANDMARK_DIM), [1, SEQUENCE_LENGTH, LANDMARK_DIM]);
      await sessionTarget.run({ input: dummyInput });
      await sessionDraft.run({ input: dummyInput });

      dummyInput.dispose();

      isONNXReady = true;
      mercyHaptic.playPattern('cosmicHarmony', currentValence.get());
      console.log(`[ONNXGestureEngine] Quantized ONNX models loaded with provider: ${currentProvider}`);
    } catch (e) {
      console.error("[ONNXGestureEngine] Activation failed", e);
      mercyHaptic.playPattern('warningPulse', 0.7);
    }
  }

  static async runInference(inputTensor: ort.Tensor): Promise<{ gesture: string; confidence: number; futureValence: number[] }> {
    if (!isONNXReady || !sessionTarget || !sessionDraft) {
      throw new Error("ONNX engine not ready");
    }

    const valence = currentValence.get();

    // Draft phase
    const draftFeeds = { input: inputTensor };
    const draftResults = await sessionDraft.run(draftFeeds);
    const draftProbs = draftResults.output.data as Float32Array;

    const draftToken = draftProbs.indexOf(Math.max(...draftProbs));

    // Verification phase
    const targetFeeds = { input: inputTensor };
    const targetResults = await sessionTarget.run(targetFeeds);
    const targetProbs = targetResults.gesture.data as Float32Array;
    const futureValenceData = targetResults.future_valence.data as Float32Array;

    const maxIdx = targetProbs.indexOf(Math.max(...targetProbs));
    const confidence = targetProbs[maxIdx];

    const gestureMap = ['none', 'pinch', 'spiral', 'figure8'];
    const gesture = confidence > 0.75 ? gestureMap[maxIdx] : 'none';

    return {
      gesture,
      confidence,
      futureValence: Array.from(futureValenceData),
    };
  }

  static isActive(): boolean {
    return isONNXReady;
  }

  static getCurrentProvider(): string {
    return currentProvider;
  }
}

export default ONNXGestureEngine;      mercyHaptic.playPattern('warningPulse', 0.7);
    }
  }

  static async runInference(inputTensor: ort.Tensor): Promise<{ gesture: string; confidence: number; futureValence: number[] }> {
    if (!isONNXReady || !sessionTarget || !sessionDraft) {
      throw new Error("ONNX engine not ready");
    }

    const valence = currentValence.get();

    const draftFeeds = { input: inputTensor };
    const draftResults = await sessionDraft.run(draftFeeds);
    const draftProbs = draftResults.output.data as Float32Array;

    const draftToken = draftProbs.indexOf(Math.max(...draftProbs));

    const targetFeeds = { input: inputTensor };
    const targetResults = await sessionTarget.run(targetFeeds);
    const targetProbs = targetResults.gesture.data as Float32Array;
    const futureValenceData = targetResults.future_valence.data as Float32Array;

    const maxIdx = targetProbs.indexOf(Math.max(...targetProbs));
    const confidence = targetProbs[maxIdx];

    const gestureMap = ['none', 'pinch', 'spiral', 'figure8'];
    const gesture = confidence > 0.75 ? gestureMap[maxIdx] : 'none';

    return {
      gesture,
      confidence,
      futureValence: Array.from(futureValenceData),
    };
  }

  static isActive(): boolean {
    return isONNXReady;
  }

  static getCurrentProvider(): string {
    return currentProvider;
  }
}

export default ONNXGestureEngine;
