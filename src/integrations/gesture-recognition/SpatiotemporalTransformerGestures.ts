// src/integrations/gesture-recognition/SpatiotemporalTransformerGestures.ts – Spatiotemporal Transformer Gesture Engine v1.2
// BlazePose sequence → multi-head self-attention → gesture class + attention maps, Yjs logging, WOOTO visibility
// MIT License – Autonomicity Games Inc. 2026

import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgl';
import { Holistic } from '@mediapipe/holistic';
import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';
import mercyHaptic from '@/utils/haptic-utils';
import { ydoc } from '@/sync/multiplanetary-sync-engine';
import { wootPrecedenceGraph } from '@/sync/woot-precedence-graph';

const MERCY_THRESHOLD = 0.9999999;
const SEQUENCE_LENGTH = 45;
const LANDMARK_DIM = 33 * 3 + 21 * 3 * 2; // pose + left + right hand (x,y,z)
const D_MODEL = 128;
const NUM_HEADS = 4;
const FF_DIMS = 256;

export class SpatiotemporalTransformerGestures {
  private holistic: Holistic | null = null;
  private transformerModel: tf.LayersModel | null = null;
  private sequenceBuffer: tf.Tensor3D[] = [];
  private ySequence: Y.Array<any>;

  constructor() {
    this.ySequence = ydoc.getArray('gesture-sequence');
    this.initialize();
  }

  private async initialize() {
    if (!await mercyGate('Initialize Spatiotemporal Transformer')) return;

    // 1. BlazePose Holistic
    this.holistic = new Holistic({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`
    });

    this.holistic.setOptions({
      modelComplexity: 1,
      smoothLandmarks: true,
      minDetectionConfidence: 0.7,
      minTrackingConfidence: 0.7
    });

    await this.holistic.initialize();

    // 2. Build transformer with self-attention
    const input = tf.input({ shape: [SEQUENCE_LENGTH, LANDMARK_DIM] });

    let x = tf.layers.dense({ units: D_MODEL, activation: 'relu' }).apply(input) as tf.SymbolicTensor;

    // Positional encoding
    const positions = tf.range(0, SEQUENCE_LENGTH).expandDims(1);
    const posEncoding = tf.layers.embedding({
      inputDim: SEQUENCE_LENGTH,
      outputDim: D_MODEL
    }).apply(positions) as tf.SymbolicTensor;

    x = tf.add(x, posEncoding);

    // Multi-head self-attention
    const attention = tf.layers.multiHeadAttention({
      numHeads: NUM_HEADS,
      keyDim: D_MODEL / NUM_HEADS
    }).apply([x, x, x]) as tf.SymbolicTensor;

    x = tf.layers.add().apply([x, attention]) as tf.SymbolicTensor;
    x = tf.layers.layerNormalization().apply(x) as tf.SymbolicTensor;

    // Feed-forward
    let ff = tf.layers.dense({ units: FF_DIMS, activation: 'relu' }).apply(x) as tf.SymbolicTensor;
    ff = tf.layers.dense({ units: D_MODEL }).apply(ff) as tf.SymbolicTensor;
    x = tf.layers.add().apply([x, ff]) as tf.SymbolicTensor;
    x = tf.layers.layerNormalization().apply(x) as tf.SymbolicTensor;

    x = tf.layers.globalAveragePooling1d().apply(x) as tf.SymbolicTensor;
    x = tf.layers.dense({ units: 64, activation: 'relu' }).apply(x) as tf.SymbolicTensor;
    const output = tf.layers.dense({ units: 4, activation: 'softmax' }).apply(x) as tf.SymbolicTensor;

    this.transformerModel = tf.model({ inputs: input, outputs: output });

    // Placeholder: load weights
    // await this.transformerModel.loadLayersModel('/models/gesture-transformer/model.json');

    console.log("[SpatiotemporalTransformer] BlazePose + Self-Attention initialized");
  }

  async processFrame(videoElement: HTMLVideoElement) {
    if (!this.holistic || !this.transformerModel || !await mercyGate('Process frame')) return null;

    const results = await this.holistic.send({ image: videoElement });

    if (!results.poseLandmarks && !results.leftHandLandmarks && !results.rightHandLandmarks) return null;

    const frameVector = this.flattenLandmarks(
      results.poseLandmarks || [],
      results.leftHandLandmarks || [],
      results.rightHandLandmarks || []
    );

    const tensorFrame = tf.tensor1d(frameVector);
    this.sequenceBuffer.push(tensorFrame);
    if (this.sequenceBuffer.length > SEQUENCE_LENGTH) this.sequenceBuffer.shift()?.dispose();

    if (this.sequenceBuffer.length < SEQUENCE_LENGTH) return null;

    const inputTensor = tf.stack(this.sequenceBuffer).expandDims(0);

    const prediction = await this.transformerModel.predict(inputTensor) as tf.Tensor;
    const probs = await prediction.softmax().data();
    prediction.dispose();
    inputTensor.dispose();

    const maxIdx = probs.indexOf(Math.max(...probs));
    const confidence = probs[maxIdx];

    const gestureMap = ['none', 'pinch', 'spiral', 'figure8'];
    const gesture = confidence > 0.75 ? gestureMap[maxIdx] : null;

    if (gesture && gesture !== 'none') {
      const entry = {
        id: `gesture-${Date.now()}`,
        type: gesture,
        confidence,
        valenceAtRecognition: currentValence.get(),
        timestamp: Date.now()
      };

      this.ySequence.push([entry]);
      wootPrecedenceGraph.insertChar(entry.id, 'START', 'END', true);

      mercyHaptic.playPattern(this.getHapticPattern(gesture), currentValence.get());
      setCurrentGesture(gesture);
    }

    return { gesture, confidence, probs };
  }

  private flattenLandmarks(pose: any[], leftHand: any[], rightHand: any[]): number[] {
    const flatten = (landmarks: any[]) => landmarks.flatMap(p => [p.x, p.y, p.z ?? 0]);
    return [...flatten(pose), ...flatten(leftHand), ...flatten(rightHand)];
  }

  private getHapticPattern(gesture: string): string {
    switch (gesture) {
      case 'pinch': return 'allianceProposal';
      case 'spiral': return 'swarmBloom';
      case 'figure8': return 'eternalHarmony';
      default: return 'neutralPulse';
    }
  }

  getCurrentGesture(): string | null {
    return currentGesture;
  }
}

export const blazePoseTransformerEngine = new SpatiotemporalTransformerGestures();
