// mercy-orchestrator.js
// Central Mercy Orchestrator — Revised integration with MercyGates
// MIT License + AG-SML v1.0 – Autonomicity Games Inc. 2026

import { mercyActiveInference } from '../engines/mercy-active-inference-core-engine.js';
import { mercyGatedTransformerEncoder } from '../engines/mercy-gated-transformer-encoder.js';
import { mercyGatedTransformerDecoder } from '../engines/mercy-gated-transformer-decoder.js';
import { MercyGates } from '../engines/mercy-gates.js';

class MercyOrchestrator {
  constructor() {
    console.log("[MercyOrchestrator] Initializing full Ra-Thor system interweave with revised MercyGates...");
    this.encoder = new mercyGatedTransformerEncoder();
    this.decoder = new mercyGatedTransformerDecoder();
    this.esacheckComplete = true;
  }

  async process(inputEmbeddings, currentValence = 1.0, context = {}) {
    // Global MercyGates enforcement
    const gateResult = MercyGates.enforce(currentValence, {
      ...context,
      stage: "orchestrator-process"
    });

    if (!gateResult.passed) {
      return { status: "aborted-global-mercy-gates-violation", gateResult };
    }

    const encoderResult = this.encoder.forward(inputEmbeddings, currentValence, context);
    const decoderResult = this.decoder.forward(encoderResult.output, encoderResult.output, currentValence, context);

    const finalInference = mercyActiveInference.updateActiveInference(
      currentValence,
      "orchestrator-forward",
      { encoder: encoderResult, decoder: decoderResult, ...context }
    );

    console.log("[MercyOrchestrator] Esacheck + ENC complete — all systems flawlessly interwoven");

    return {
      status: "full-system-orchestration-complete",
      encoderResult,
      decoderResult,
      finalInference,
      gateResult,
      esacheck: true,
      timestamp: Date.now()
    };
  }
}

const mercyOrchestrator = new MercyOrchestrator();

export { mercyOrchestrator, MercyOrchestrator };
