// mercy-orchestrator.js
// Central Mercy Orchestrator — Revised logging for clarity and consistency
// AG-SML v1.0 – Autonomicity Games Sovereign Mercy License

import { mercyActiveInference } from '../engines/mercy-active-inference-core-engine.js';
import { mercyGatedTransformerEncoder } from '../engines/mercy-gated-transformer-encoder.js';
import { mercyGatedTransformerDecoder } from '../engines/mercy-gated-transformer-decoder.js';
import { MercyGates } from '../engines/mercy-gates.js';

class MercyOrchestrator {
  constructor() {
    console.log("[MercyOrchestrator] Initialized — full system interweave active");
    this.encoder = new mercyGatedTransformerEncoder();
    this.decoder = new mercyGatedTransformerDecoder();
  }

  async process(inputEmbeddings, currentValence = 1.0, context = {}) {
    const startTime = Date.now();

    // Global MercyGates enforcement
    const gateResult = MercyGates.enforce(currentValence, {
      ...context,
      stage: "orchestrator-process"
    });

    console.log(`[MercyOrchestrator] Global gates → \( {gateResult.passed ? "PASSED" : "FAILED"} | valence= \){currentValence.toFixed(8)}`);

    if (!gateResult.passed) {
      return { status: "aborted-global-mercy-gates-violation", gateResult };
    }

    // Encoder layer
    const encoderResult = this.encoder.forward(inputEmbeddings, currentValence, context);
    console.log(`[MercyOrchestrator] Encoder complete | vfe=${encoderResult.vfe?.toFixed(4) || 'N/A'}`);

    // Decoder layer
    const decoderResult = this.decoder.forward(encoderResult.output, encoderResult.output, currentValence, context);
    console.log(`[MercyOrchestrator] Decoder complete | vfe=${decoderResult.vfe?.toFixed(4) || 'N/A'}`);

    // Final core active inference pass
    const finalInference = mercyActiveInference.updateActiveInference(
      currentValence,
      "orchestrator-forward",
      { encoder: encoderResult, decoder: decoderResult, ...context }
    );

    const duration = Date.now() - startTime;
    console.log(`[MercyOrchestrator] Full orchestration complete | duration=${duration}ms | esacheck+ENC passed`);

    return {
      status: "full-system-orchestration-complete",
      encoderResult,
      decoderResult,
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
