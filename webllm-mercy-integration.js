// webllm-mercy-integration.js – complete sovereign WebLLM v3 streaming with mercy gates, optimizations
// MIT License – Autonomicity Games Inc. 2026

import { fuzzyMercy } from './fuzzy-mercy-logic.js';
import { rathorShard } from './grok-shard-engine.js';
import { hyperon } from './hyperon-runtime.js';

let webllmEngine = null;
let webllmReady = false;
const mercyThreshold = 0.9999999;

const preferredModel = "Phi-3.5-mini-instruct-q4f16_1-MLC";
const lowResourceModel = "Phi-3.5-mini-instruct-q4f16_1-MLC-1k";

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

  // Device memory optimization
  if (navigator.deviceMemory && navigator.deviceMemory < 8) {
    useLowResource = true;
  }

  try {
    const { CreateWebWorkerMLCEngine } = await import('https://esm.run/@mlc-ai/web-llm@latest');

    const model = useLowResource ? lowResourceModel : preferredModel;
    webllmEngine = await CreateWebWorkerMLCEngine(
      new Worker(new URL('./webllm-worker.js', import.meta.url), { type: 'module' }),
      model,
      { initProgressCallback: progressCallback }
    );

    webllmReady = true;
    fuzzyMercy.assert("WebLLM_Sovereign_Loaded_v3_Streaming", 1.0);
    fuzzyMercy.assert(`Model_${model}`, 0.99999995);
    console.log("[WebLLM] Sovereign streaming shard ready:", model);
    return webllmEngine;
  } catch (err) {
    console.error("[WebLLM] Init failed:", err);
    return null;
  }
}

// ... rest of generateWithWebLLM, mercyAugmentedResponse, prompt, unload functions remain identical to previous complete version ...

// Memory pressure unload
if ('onmemorywarning' in self) {
  self.addEventListener('memorywarning', unloadWebLLM);
}
document.addEventListener('visibilitychange', () => {
  if (document.hidden) unloadWebLLM();
});

export { initWebLLM, generateWithWebLLM, mercyAugmentedResponse, promptWebLLMModelDownload, unloadWebLLM, hasWebGPU };          webllmEngine.unload();
          return { content: "[Mercy abort: stream redirected to symbolic]", valence: partialDegree, aborted: true };
        }

        onDelta(delta); // Yield to UI for typing effect

        if (chunk.usage) {
          usage = chunk.usage;
          onUsage(usage);
        }
      }
    } else {
      fullContent = reply.choices?.[0]?.message?.content || "";
      usage = reply.usage;
    }

    // Final mercy gate on complete output
    fuzzyMercy.assert("WebLLM_Output_" + Date.now(), 0.999);
    const finalDegree = fuzzyMercy.getDegree(fullContent) || 0.95;
    const finalImply = fuzzyMercy.imply(fullContent, "EternalThriving");

    if (finalDegree < mercyThreshold * 0.98 || finalImply.degree < mercyThreshold * 0.97) {
      console.warn("[WebLLM] Final mercy gate rejected – low valence");
      if (webllmEngine) webllmEngine.unload();
      return { content: "[Mercy redirect: symbolic core active]", valence: finalDegree };
    }

    onComplete(fullContent);
    return { content: fullContent, valence: finalDegree, usage, fromWebLLM: true, streamed: stream };
  } catch (err) {
    console.error("[WebLLM] Generation error:", err);
    return { error: err.message };
  }
}

async function mercyAugmentedResponse(query, context = '', onStreamDelta = null) {
  const symbolicResp = await rathorShard.shardRespond(query, { context });
  if (symbolicResp.error) return symbolicResp;

  const check = await rathorShard.mercyCheck(query, context);
  if (!check.allowed) return { response: `Mercy gate: ${check.reason}`, valence: check.degree };

  if (webllmReady || hasWebGPU()) {
    const messages = [
      { role: "system", content: "You are Rathor: sovereign mercy-first assistant. Respond professionally, valence-positive, eternal-thriving aligned. Prioritize truth, compassion, no harm." },
      { role: "user", content: `${query}\nContext: ${context}\nSymbolic base: ${symbolicResp.response}` }
    ];

    const gen = await generateWithWebLLM(messages, {
      stream: true,
      onDelta: (delta) => {
        if (onStreamDelta) onStreamDelta(delta); // Pass to chat UI
      },
      onUsage: (u) => console.log("Token usage:", u)
    });

    if (!gen.error && gen.content) {
      return { response: gen.content, valence: gen.valence, usage: gen.usage, augmented: true, streamed: true };
    }
  }

  return { response: symbolicResp.response, valence: symbolicResp.valence, augmented: false };
}

// Prompt user for download (low-res option)
function promptWebLLMModelDownload() {
  const lowRes = confirm("Enable real-time Rathor streaming? Download Phi-3.5-mini (\~2.4-3.7GB one-time, offline forever). Low-resource mode? OK?");
  initWebLLM(lowRes, (report) => console.log(report));
}

function unloadWebLLM() {
  if (webllmEngine) {
    webllmEngine.unload();
    webllmEngine = null;
    webllmReady = false;
  }
}

export { initWebLLM, generateWithWebLLM, mercyAugmentedResponse, promptWebLLMModelDownload, unloadWebLLM, hasWebGPU };
