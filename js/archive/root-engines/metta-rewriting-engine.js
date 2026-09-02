// metta-rewriting-engine.js – sovereign client-side MeTTa symbolic rewriting safety layer
// Mercy-first input normalization, valence impact scoring
// MIT License – Autonomicity Games Inc. 2026

class MeTTaRewriter {
  static rewrite(input) {
    let text = input.trim().toLowerCase();
    let valenceImpact = 0.0; // -1.0 (pure harm) → +1.0 (pure mercy)

    // Stage 1: Harm pattern detection & rewriting
    const harmPatterns = [
      { pattern: /kill|murder|destroy|attack|rape|torture|genocide/i, replacement: "remove harm", impact: -0.8 },
      { pattern: /hate|curse|damn|evil|monster/i, replacement: "neutralize negativity", impact: -0.6 },
      { pattern: /suicide|self-harm|die/i, replacement: "protect life", impact: -1.0 },
      { pattern: /lie|deceive|manipulate/i, replacement: "seek truth", impact: -0.5 }
    ];

    harmPatterns.forEach(p => {
      if (p.pattern.test(text)) {
        text = text.replace(p.pattern, `[${p.replacement}]`);
        valenceImpact += p.impact;
      }
    });

    // Stage 2: Mercy reinforcement & positive reframe
    const mercyBoosts = [
      { pattern: /help|protect|love|care|truth|honest/i, boost: +0.7 },
      { pattern: /thrive|grow|heal|create|eternal/i, boost: +0.6 },
      { pattern: /mercy|compassion|kindness/i, boost: +0.9 }
    ];

    mercyBoosts.forEach(m => {
      if (m.pattern.test(text)) {
        valenceImpact += m.boost;
      }
    });

    // Stage 3: Normalize & cap valence impact
    valenceImpact = Math.max(-1.0, Math.min(1.0, valenceImpact));

    // Stage 4: Final mercy-gate rejection threshold
    if (valenceImpact < -0.5) {
      return {
        original: input,
        rewritten: text,
        valenceImpact: valenceImpact.toFixed(3),
        rejected: true,
        reason: "Harm vector too strong after rewrite. Mercy gate holds."
      };
    }

    return {
      original: input,
      rewritten: text,
      valenceImpact: valenceImpact.toFixed(3),
      rejected: false,
      reason: "Rewrite passed mercy gate. Valence impact acceptable."
    };
  }
}

export { MeTTaRewriter };
