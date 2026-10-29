// Rathor™ Mercy-Transformers Integration — Local Model Augmentation
// User guidance injected, offline-first

import { pipeline } from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2";

let textGenPipeline = null;
let transformersReady = false;
let userGuidanceCache = null;

async function initTransformers() {
  if (transformersReady) return;
  try {
    textGenPipeline = await pipeline('text-generation', 'Xenova/Phi-3-mini-4k-instruct');
    transformersReady = true;
    console.log("[Mercy] Transformers pipeline ready");
  } catch (err) {
    console.warn("[Mercy] Transformers init failed", err);
  }
}

async function getUserGuidance() {
  if (userGuidanceCache === null) {
    try {
      const stored = await rathorDB.get('settings', 'userPrompt');
      userGuidanceCache = stored?.value?.trim() || '';
    } catch (err) {
      userGuidanceCache = '';
    }
  }
  return userGuidanceCache;
}

async function mercyAugmentedResponse(query, context = '') {
  const symbolicResp = await rathorShard.shardRespond(query, { context });
  if (symbolicResp.error) return symbolicResp;

  const check = await rathorShard.mercyCheck(query + context);
  if (!check.allowed) {
    return { response: `Mercy holds the gate: ${check.reason}`, valence: 0.9999999 };
  }

  const userGuidance = await getUserGuidance();

  let prompt = `${query}\nSymbolic base: ${symbolicResp.response}\nContext: ${context}`;
  if (userGuidance) {
    prompt = `Eternal User Guidance (follow in all responses):\n${userGuidance}\n\n${prompt}`;
  }

  if (transformersReady && textGenPipeline) {
    try {
      const output = await textGenPipeline(prompt, {
        max_new_tokens: 1024,
        temperature: 0.7,
        do_sample: true
      });
      return { response: output[0].generated_text, valence: 0.999999, augmented: true };
    } catch (err) {
      console.warn("[Mercy] Transformers generation failed", err);
    }
  }

  // Fallback
  let response = symbolicResp.response;
  if (userGuidance) {
    response = `Guided by Eternal Guidance ⚡️\n${userGuidance}\n\n${response}`;
  }
  return { response, valence: symbolicResp.valence || 0.999999, augmented: false };
}

// Auto-init
document.addEventListener('DOMContentLoaded', initTransformers);

export { mercyAugmentedResponse, initTransformers };
