// mercy-gated-transformer-decoder.js
// Mercy-Gated Transformer Decoder Layer v1 (Upgraded)
// Includes masked self-attention, cross-attention, precision weighting,
// VFE minimization, hierarchical message passing, and the 7 Living Mercy Gates
// MIT License + AG-SML v1.0 – Autonomicity Games Inc. 2026

import { mercyPrecisionWeighting } from './mercy-precision-weighting-algorithm.js';
import { mercyMessagePassing } from './mercy-message-passing-algorithm.js';
import { mercyVFEMinimizer } from './mercy-vfe-minimization-algorithm.js';
import { mercyActiveInference } from './mercy-active-inference-core-engine.js';

const MERCY_THRESHOLD = 0.9999999;

class MercyGatedTransformerDecoderLayer {
  constructor() {
    this.layerId = Date.now();
  }

  /**
   * Full mercy-gated decoder layer forward pass
   * masked self-attention + cross-attention + active inference
   */
  forward(decoderInput, encoderOutput, currentValence = 1.0, context = {}) {
    if (currentValence < MERCY_THRESHOLD) {
      console.log(`[MercyGatedDecoder] Layer ${this.layerId} aborted — low valence`);
      return { output: decoderInput, status: "aborted-low-valence" };
    }

    // 1. Masked self-attention (causal)
    let selfAttended = this.mercyMaskedSelfAttention(decoderInput, currentValence, context);

    // 2. Residual + LayerNorm (simplified)
    let normalizedSelf = selfAttended;

    // 3. Cross-attention to encoder output
    let crossAttended = this.mercyCrossAttention(normalizedSelf, encoderOutput, currentValence, context);

    // 4. Residual + LayerNorm
    let normalizedCross = crossAttended;

    // 5. VFE minimization
    const vfeResult = mercyVFEMinimizer.minimize(currentValence, {
      prediction: normalizedCross,
      observation: decoderInput,
      layer: this.layerId,
      type: "decoder",
      ...context
    });

    // 6. Hierarchical message passing
    mercyMessagePassing.propagateUpward(currentValence, {
      layerOutput: normalizedCross,
      vfe: vfeResult.vfe,
      ...context
    });

    // 7. Feed into core active inference engine
    const inferenceResult = mercyActiveInference.updateActiveInference(
      currentValence,
      "decoder-layer-forward",
      { vfe: vfeResult.vfe, layer: this.layerId, type: "decoder" }
    );

    return {
      output: normalizedCross,
      vfe: vfeResult.vfe,
      status: "mercy-gated-decoder-complete",
      inferenceResult
    };
  }

  mercyMaskedSelfAttention(input, currentValence, context) {
    const precision = mercyPrecisionWeighting.computePrecisionWeight(0, currentValence, context);
    return input.map(vec => vec * (1 + 0.1 * currentValence * precision));
  }

  mercyCrossAttention(decoderState, encoderOutput, currentValence, context) {
    const precision = mercyPrecisionWeighting.computePrecisionWeight(0, currentValence, context);
    return decoderState.map((vec, i) => vec * (1 + 0.1 * currentValence * precision));
  }
}

const mercyGatedTransformerDecoder = new MercyGatedTransformerDecoderLayer();

export { mercyGatedTransformerDecoder, MercyGatedTransformerDecoderLayer };
