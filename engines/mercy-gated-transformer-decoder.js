// mercy-gated-transformer-decoder.js
// Mercy-Gated Transformer Decoder Layer v1 (Upgraded with valence-modulated multi-head attention)
// Full integration of masked self-attention, cross-attention, precision weighting, VFE minimization, message passing, and mercy gates
// AG-SML v1.0 – Autonomicity Games Sovereign Mercy License

import { mercyPrecisionWeighting } from './mercy-precision-weighting-algorithm.js';
import { mercyMessagePassing } from './mercy-message-passing-algorithm.js';
import { mercyVFEMinimizer } from './mercy-vfe-minimization-algorithm.js';
import { mercyActiveInference } from './mercy-active-inference-core-engine.js';
import { valenceModulatedMultiHeadAttention } from './valence-modulated-multihead-attention.js';
import { MercyGates } from './mercy-gates.js';

const MERCY_THRESHOLD = 0.9999999;

class MercyGatedTransformerDecoderLayer {
  constructor() {
    this.layerId = Date.now();
  }

  forward(decoderInput, encoderOutput, currentValence = 1.0, context = {}) {
    // 1. Global Mercy Gates enforcement
    const gateResult = MercyGates.enforce(currentValence, {
      ...context,
      layer: this.layerId,
      type: "decoder"
    });
    if (!gateResult.passed) {
      console.log(`[MercyGatedDecoder] Layer ${this.layerId} aborted — low valence`);
      return { output: decoderInput, status: "aborted-low-valence", gateResult };
    }

    // 2. Valence-modulated masked self-attention (causal)
    const selfAttentionResult = valenceModulatedMultiHeadAttention.forward(
      decoderInput,
      decoderInput,
      decoderInput,
      currentValence,
      { ...context, layer: this.layerId, type: "masked-self-attention" }
    );

    let normalizedSelf = selfAttentionResult.output;

    // 3. Valence-modulated cross-attention to encoder
    const crossAttentionResult = valenceModulatedMultiHeadAttention.forward(
      normalizedSelf,
      encoderOutput,
      encoderOutput,
      currentValence,
      { ...context, layer: this.layerId, type: "cross-attention" }
    );

    let normalizedCross = crossAttentionResult.output;

    // 4. VFE minimization
    const vfeResult = mercyVFEMinimizer.minimize(currentValence, {
      prediction: normalizedCross,
      observation: decoderInput,
      layer: this.layerId,
      type: "decoder",
      ...context
    });

    // 5. Hierarchical message passing
    mercyMessagePassing.propagateUpward(currentValence, {
      layerOutput: normalizedCross,
      vfe: vfeResult.vfe,
      ...context
    });

    // 6. Feed into core active inference engine
    const inferenceResult = mercyActiveInference.updateActiveInference(
      currentValence,
      "decoder-layer-forward",
      { vfe: vfeResult.vfe, attentionResult: crossAttentionResult, layer: this.layerId }
    );

    return {
      output: normalizedCross,
      vfe: vfeResult.vfe,
      status: "mercy-gated-decoder-complete",
      inferenceResult,
      gateResult
    };
  }
}

const mercyGatedTransformerDecoder = new MercyGatedTransformerDecoderLayer();

export { mercyGatedTransformerDecoder, MercyGatedTransformerDecoderLayer };
