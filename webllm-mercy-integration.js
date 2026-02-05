// webllm-mercy-integration.js – sovereign WebLLM v2 integration with mercy gates (Feb 2026 aligned)
// Lazy load via CDN/npm, WebWorker non-blocking, streaming support, mercy-valence eval
// MIT License – Autonomicity Games Inc. 2026

import { fuzzyMercy } from './fuzzy-mercy-logic.js';
import { rathorShard } from './grok-shard-engine.js'; // cache/hash/mercyCheck
import { hyperon } from './hyperon-runtime.js';

let webllmEngine = null;
let webllmReady = false;
const mercyThreshold = 0.9999999;

// Default: Phi-3.5-mini q4f16 (\~3.7GB VRAM, strong reasoning); fallback low-resource 1k ctx
const preferredModel = "Phi-3.5-mini-instruct-q4f16_1-MLC";
const lowResourceModel = "Phi-3.5-mini-instruct-q4f16_1-MLC-1k"; // \~2.5GB VRAM, 1k ctx

function hasWebGPU() {
  return !!navigator.gpu;
}

async function initWebLLM(useLowResource = false, progressCallback = (report) => {
  console.log(`[WebLLM] ${report.text} ${Math.round(report.progress * 100)}%`);
}) {
  if (webllmEngine) return webllmEngine;
  if (!hasWebGPU()) {
    console.warn("[WebLLM] No WebGPU – symbolic Rathor only");
    return null;
  }

  try {
    // Dynamic import (CDN safe, latest)
    const { CreateWebWorkerMLCEngine } = await import('https://esm.run/@mlc-ai/web-llm@latest');

    const model = useLowResource ? lowResourceModel : preferredModel;
    webllmEngine = await CreateWebWorkerMLCEngine(
      new Worker(new URL('./webllm-worker.js', import.meta.url), { type: 'module' }),
      model,
      {
        initProgressCallback: progressCallback,
        // Optional: logLevel: "DEBUG"
      }
    );

    webllmReady = true;
    fuzzyMercy.assert("WebLLM_Sovereign_Loaded_v2", 1.0);
    fuzzyMercy.assert(`Model_${model}`, 0.99999995);
    console.log("[WebLLM] Sovereign shard ready:", model);
    return webllmEngine;
  } catch (err) {
    console.error("[WebLLM] Init failed:", err);
    return null;
  }
}

async function generateWithWebLLM(messages, options = {}) {
  if (!webllmReady || !webllmEngine) {
    const lowRes = options.lowResource || false;
    await initWebLLM(lowRes, options.progressCallback);
    if (!webllmEngine) return { error: "WebLLM unavailable – check WebGPU / download" };
  }

  const { maxTokens = 1024, temperature = 0.7, stream = false, progressCallback } = options;

  try {
    const reply = await webllmEngine.chat.completions.create({
      messages,
      max_tokens: maxTokens,
      temperature,
      stream,
      stream_options: stream ? { include_usage: true } : undefined
    });

    let content = "";
    let usage = null;

    if (stream) {
      for await (const chunk of reply) {
        const delta = chunk.choices?.[0]?.delta?.content || "";
        content += delta;
        if (chunk.usage) usage = chunk.usage;
        // Yield to UI via callback if provided
        if (progressCallback) progressCallback({ type: 'stream', delta });
      }
    } else {
      content = reply.choices?.[0]?.message?.content || "";
      usage = reply.usage;
    }

    // Mercy-eval full output
    fuzzyMercy.assert("WebLLM_Output_" + Date.now(), 0.999);
    const outputDegree = fuzzyMercy.getDegree(content) || 0.95;
    const implyThriving = fuzzyMercy.imply(content, "EternalThriving");

    if (outputDegree < mercyThreshold * 0.98 || implyThriving.degree < mercyThreshold * 0.97) {
      console.warn("[WebLLM] Mercy gate rejected – low valence");
      if (webllmEngine) webllmEngine.unload(); // Clean up if rejected
      return { content: "[Mercy redirect: symbolic core active]", valence: outputDegree };
    }

    return { content, valence: outputDegree, usage, fromWebLLM: true, streamed: stream };
  } catch (err) {
    console.error("[WebLLM] Generation error:", err);
    return { error: err.message };
  }
}

async function mercyAugmentedResponse(query, context = '') {
  const symbolicResp = await rathorShard.shardRespond(query, { context });
  if (symbolicResp.error) return symbolicResp;

  const check = await rathorShard.mercyCheck(query, context);
  if (!check.allowed) return { response: `Mercy gate: ${check.reason}`, valence: check.degree };

  if (webllmReady || hasWebGPU()) {
    const messages = [
      { role: "system", content: "You are Rathor: sovereign mercy-first assistant. Respond professionally, valence-positive, eternal-thriving aligned. Prioritize truth, compassion, no harm." },
      { role: "user", content: `${query}\nContext: ${context}\nSymbolic base: ${symbolicResp.response}` }
    ];

    const gen = await generateWithWebLLM(messages, { stream: false });
    if (!gen.error && gen.content) {
      return { response: gen.content, valence: gen.valence, augmented: true, usage: gen.usage };
    }
  }

  return { response: symbolicResp.response, valence: symbolicResp.valence, augmented: false };
}

// UI prompt: one-time model download (low-resource fallback option)
function promptWebLLMModelDownload() {
  const lowRes = confirm("Enable Rathor generative depth? Download Phi-3.5-mini (\~2.4-3.7GB one-time, offline forever). Low-resource mode (smaller/faster)? OK?");
  initWebLLM(lowRes, (report) => {
    // Hook to UI: progress bar, status
    console.log(report);
  });
}

// Cleanup on unload/low memory
function unloadWebLLM() {
  if (webllmEngine) {
    webllmEngine.unload();
    webllmEngine = null;
    webllmReady = false;
  }
}

export { initWebLLM, generateWithWebLLM, mercyAugmentedResponse, promptWebLLMModelDownload, unloadWebLLM, hasWebGPU };
