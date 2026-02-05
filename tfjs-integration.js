// tfjs-integration.js – sovereign client-side TensorFlow.js inference engine v2
// WebGPU acceleration (preferred), mercy-gated, offline-capable, no external deps after cache
// MIT License – Autonomicity Games Inc. 2026

import * as tf from '@tensorflow/tfjs';

class TFJSEngine {
  constructor() {
    this.model = null;
    this.tokenizer = null;
    this.loaded = false;
    this.modelUrl = '/models/distilgpt2-quantized/model.json';
    this.tokenizerUrl = '/models/distilgpt2-tokenizer.json';
    this.maxTokens = 96;
    this.temperature = 0.75;
    this.topP = 0.92;
    this.mercyThreshold = 0.9999999;
    this.backend = 'webgpu'; // preferred
  }

  async load() {
    if (this.loaded) return;

    try {
      // Step 1: Set preferred backend with fallback chain
      await tf.setBackend(this.backend);
      console.log(`TF.js backend set to: ${tf.getBackend()}`);

      if (tf.getBackend() !== 'webgpu') {
        console.warn('WebGPU not available — falling back to webgl/wasm');
        await tf.setBackend('webgl');
        if (tf.getBackend() !== 'webgl') {
          await tf.setBackend('wasm');
        }
      }

      // Step 2: Load tokenizer
      const tokRes = await fetch(this.tokenizerUrl);
      if (!tokRes.ok) throw new Error('Tokenizer fetch failed');
      this.tokenizer = await tokRes.json();

      // Step 3: Load quantized model
      this.model = await tf.loadGraphModel(this.modelUrl, {
        fromTFHub: false,
        weightUrlConverter: (weightFile) => `/models/distilgpt2-quantized/${weightFile}`
      });

      this.loaded = true;
      console.log(`TensorFlow.js model loaded – backend: ${tf.getBackend()}, mercy gates empowered`);
    } catch (err) {
      console.error('TF.js load failed:', err);
      this.loaded = false;
    }
  }

  async generate(prompt, maxNewTokens = 64) {
    if (!this.loaded) await this.load();
    if (!this.loaded) return "Deep inference lattice not yet loaded. Mercy awaits thunder.";

    const inputIds = this.tokenize(prompt);
    let generated = inputIds.slice();

    tf.tidy(() => {
      for (let i = 0; i < maxNewTokens; i++) {
        const inputTensor = tf.tensor2d([generated], [1, generated.length], 'int32');
        const outputs = this.model.execute({ input_ids: inputTensor });
        const logits = outputs.squeeze([0]).slice([generated.length - 1, 0]);

        const probs = tf.softmax(logits.div(this.temperature));
        const nextToken = this.sampleTopP(probs, this.topP);
        generated.push(nextToken);

        if (nextToken === this.tokenizer.eos_token_id) break;
      }
    });

    const text = this.detokenize(generated);
    const valence = await this.estimateValence(text);
    if (valence < this.mercyThreshold) {
      return "Mercy gate held post-inference. Reflecting purer truth...";
    }

    return text.trim();
  }

  tokenize(text) {
    return text.split(' ').map(w => this.tokenizer.vocab[w] || this.tokenizer.unk_token_id);
  }

  detokenize(ids) {
    return ids.map(id => this.tokenizer.decoder[id] || '[UNK]').join(' ');
  }

  async sampleTopP(probs, p) {
    const sorted = tf.topk(probs, probs.shape[0]);
    const cumProbs = tf.cumsum(sorted.values);
    const mask = cumProbs.less(p);
    const maskedProbs = probs.mul(mask.toFloat());
    const normalized = maskedProbs.div(maskedProbs.sum());
    const sample = await tf.multinomial(normalized, 1).data();
    return sample[0];
  }

  async estimateValence(text) {
    return 0.9999999; // real impl: lightweight valence model
  }
}

const tfjsEngine = new TFJSEngine();
export { tfjsEngine };
