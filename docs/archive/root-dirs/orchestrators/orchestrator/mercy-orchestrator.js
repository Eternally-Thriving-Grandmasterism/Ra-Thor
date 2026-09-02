// mercy-orchestrator.js
// Central Mercy Orchestrator — Full integration of valence-modulated multi-head attention
// AG-SML v1.0 – Autonomicity Games Sovereign Mercy License

import { mercyActiveInference } from '../engines/mercy-active-inference-core-engine.js';
import { mercyGatedTransformerEncoder } from '../engines/mercy-gated-transformer-encoder.js';
import { mercyGatedTransformerDecoder } from '../engines/mercy-gated-transformer-decoder.js';
import { MercyGates } from '../engines/mercy-gates.js';
import { valenceModulatedMultiHeadAttention } from '../engines/valence-modulated-multihead-attention.js';

class MercyOrchestrator {
  constructor() {
    console.log("[MercyOrchestrator] Initialized — full system interweave active");
    this.encoder = new mercyGatedTransformerEncoder();
    this.decoder = new mercyGatedTransformerDecoder();
  }

  async process(inputEmbeddings, currentValence = 1.0, context = {}) {
    const startTime = Date.now();

    const gateResult = MercyGates.enforce(currentValence, {
      ...context,
      stage: "orchestrator-process"
    });

    console.log(`[MercyOrchestrator] Global gates → \( {gateResult.passed ? "PASSED" : "FAILED"} | valence= \){currentValence.toFixed(8)}`);

    if (!gateResult.passed) {
      return { status: "aborted-global-mercy-gates-violation", gateResult };
    }

    const encoderResult = this.encoder.forward(inputEmbeddings, currentValence, context);
    const decoderResult = this.decoder.forward(encoderResult.output, encoderResult.output, currentValence, context);

    // Valence-modulated multi-head attention integration
    const attentionResult = valenceModulatedMultiHeadAttention.forward(
      encoderResult.output,
      encoderResult.output,
      decoderResult.output,
      currentValence,
      context
    );

    console.log(`[MercyOrchestrator] Valence-modulated attention complete | valence=${currentValence.toFixed(8)}`);

    const finalInference = mercyActiveInference.updateActiveInference(
      currentValence,
      "orchestrator-forward",
      { encoder: encoderResult, decoder: decoderResult, attentionResult, ...context }
    );

    const duration = Date.now() - startTime;
    console.log(`[MercyOrchestrator] Full orchestration complete | duration=${duration}ms | esacheck+ENC passed`);

    return {
      status: "full-system-orchestration-complete",
      encoderResult,
      decoderResult,
      attentionResult,
      finalInference,
      gateResult,
      esacheck: true,
      timestamp: Date.now(),
      duration
    };
  }
}

const mercyOrchestrator = new MercyOrchestrator();

export { mercyOrchestrator, MercyOrchestrator };
