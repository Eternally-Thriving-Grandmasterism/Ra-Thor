/**
 * live_frame_bridge.js
 * Ra-Thor Live Frame Bridge — Browser / JS Glue v1.1
 *
 * Now fully wired to the wasm-bindgen LiveVisionBridge.
 *
 * Path:
 *   getUserMedia
 *     → MediaStreamTrackProcessor
 *       → VideoFrame
 *         → luma conversion
 *           → LumaRing (prev / curr)
 *             → LiveVisionBridge.perceive_from_luma_pair (wasm)
 *               → CommonFateResult
 *
 * TOLC 8 Mercy Gated | PATSAGi Visual Council | ONE Organism
 * AG-SML v1.0 | Eternally-Thriving-Grandmasterism 2026
 */

export class LiveFrameBridge {
  /**
   * @param {GPUDevice} device
   * @param {object}    options
   * @param {number}    [options.width=640]
   * @param {number}    [options.height=360]
   * @param {number}    [options.frameRate=30]
   * @param {number}    [options.lumaMode=0]   0=BT.709, 1=average, 2=BT.601
   * @param {object}    [options.wasmBridge]  instance of LiveVisionBridge from wasm-bindgen
   * @param {number}    [options.valence=1.0]
   * @param {boolean}   [options.ghostFont=false]
   * @param {function}  [options.onResult]    called with the CommonFateResult object
   * @param {function}  [options.onLumaPair]  optional raw pair callback
   * @param {function}  [options.onError]
   */
  constructor(device, options = {}) {
    this.device = device;
    this.width = options.width ?? 640;
    this.height = options.height ?? 360;
    this.frameRate = options.frameRate ?? 30;
    this.lumaMode = options.lumaMode ?? 0;
    this.wasmBridge = options.wasmBridge ?? null;
    this.valence = options.valence ?? 1.0;
    this.ghostFont = options.ghostFont ?? false;
    this.onResult = options.onResult ?? null;
    this.onLumaPair = options.onLumaPair ?? null;
    this.onError = options.onError ?? console.error;

    this.stream = null;
    this.track = null;
    this.processor = null;
    this.reader = null;
    this.running = false;

    this.prevLuma = null;
    this.currLuma = null;
    this.frameCount = 0;
  }

  async start() {
    if (this.running) return;

    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          width: { ideal: this.width },
          height: { ideal: this.height },
          frameRate: { ideal: this.frameRate },
          facingMode: 'user',
        },
      });

      this.track = this.stream.getVideoTracks()[0];
      const settings = this.track.getSettings();
      this.width = settings.width || this.width;
      this.height = settings.height || this.height;

      if (typeof MediaStreamTrackProcessor !== 'undefined') {
        this.processor = new MediaStreamTrackProcessor({ track: this.track });
        this.reader = this.processor.readable.getReader();
        this.running = true;
        this._pump();
      } else {
        await this._startVideoElementFallback();
      }

      console.log(`[LiveFrameBridge] Camera started ${this.width}×${this.height}`);
    } catch (err) {
      this.onError(err);
      throw err;
    }
  }

  async stop() {
    this.running = false;
    if (this.reader) {
      try { await this.reader.cancel(); } catch (_) {}
      this.reader = null;
    }
    this.processor = null;
    if (this.track) { this.track.stop(); this.track = null; }
    if (this.stream) {
      this.stream.getTracks().forEach(t => t.stop());
      this.stream = null;
    }
    this.prevLuma = null;
    this.currLuma = null;
    this.frameCount = 0;
    console.log('[LiveFrameBridge] Stopped');
  }

  async _pump() {
    while (this.running && this.reader) {
      let result;
      try {
        result = await this.reader.read();
      } catch (err) {
        if (this.running) this.onError(err);
        break;
      }
      if (result.done) break;

      const frame = result.value;
      try {
        await this._ingestVideoFrame(frame);
      } catch (err) {
        this.onError(err);
      } finally {
        frame.close();
      }
    }
  }

  async _ingestVideoFrame(frame) {
    const width = frame.displayWidth || frame.codedWidth || this.width;
    const height = frame.displayHeight || frame.codedHeight || this.height;
    const timestamp = frame.timestamp ?? performance.now() * 1000;

    const lumaData = await this._convertViaCPU(frame, width, height);

    this.prevLuma = this.currLuma;
    this.currLuma = { data: lumaData, width, height, timestamp };
    this.frameCount += 1;

    if (this.frameCount < 2 || !this.prevLuma || !this.currLuma) return;

    const pair = {
      prev: this.prevLuma,
      curr: this.currLuma,
      width,
      height,
      timestamp,
      frameCount: this.frameCount,
    };

    if (this.onLumaPair) {
      this.onLumaPair(pair);
    }

    // Final thin layer: push into the wasm vision bridge
    if (this.wasmBridge && typeof this.wasmBridge.perceive_from_luma_pair === 'function') {
      try {
        const result = await this.wasmBridge.perceive_from_luma_pair(
          this.prevLuma.data,
          this.currLuma.data,
          width,
          height,
          this.valence,
          this.ghostFont,
        );
        if (this.onResult) {
          this.onResult(result);
        }
      } catch (err) {
        this.onError(err);
      }
    }
  }

  async _convertViaCPU(frame, width, height) {
    const size = frame.allocationSize({ format: 'RGBA' });
    const buffer = new ArrayBuffer(size);
    const layout = { offset: 0, bytesPerRow: width * 4, rowsPerImage: height };
    await frame.copyTo(buffer, { layout, format: 'RGBA' });

    const rgba = new Uint8ClampedArray(buffer);
    const luma = new Float32Array(width * height);

    const rC = this.lumaMode === 2 ? 0.299 : 0.2126;
    const gC = this.lumaMode === 2 ? 0.587 : 0.7152;
    const bC = this.lumaMode === 2 ? 0.114 : 0.0722;

    for (let i = 0, p = 0; i < luma.length; i++, p += 4) {
      if (this.lumaMode === 1) {
        luma[i] = (rgba[p] + rgba[p + 1] + rgba[p + 2]) / (3 * 255);
      } else {
        luma[i] = (rgba[p] * rC + rgba[p + 1] * gC + rgba[p + 2] * bC) / 255;
      }
    }
    return luma;
  }

  async _startVideoElementFallback() {
    const video = document.createElement('video');
    video.srcObject = this.stream;
    video.muted = true;
    video.playsInline = true;
    await video.play();
    this.running = true;

    const onFrame = async () => {
      if (!this.running) return;
      if (typeof VideoFrame !== 'undefined') {
        const frame = new VideoFrame(video, { timestamp: performance.now() * 1000 });
        try {
          await this._ingestVideoFrame(frame);
        } finally {
          frame.close();
        }
      }
      if (this.running) {
        video.requestVideoFrameCallback
          ? video.requestVideoFrameCallback(onFrame)
          : requestAnimationFrame(onFrame);
      }
    };

    if (video.requestVideoFrameCallback) {
      video.requestVideoFrameCallback(onFrame);
    } else {
      requestAnimationFrame(onFrame);
    }
  }
}

/**
 * Convenience factory.
 *
 * Full end-to-end example:
 *
 *   import init, { LiveVisionBridge } from './pkg/ra_thor.js';
 *   import { createLiveFrameBridge } from './js/live_frame_bridge.js';
 *
 *   await init();
 *   const wasmBridge = new LiveVisionBridge();
 *
 *   const bridge = await createLiveFrameBridge(device, {
 *     wasmBridge,
 *     onResult: (result) => {
 *       console.log('Common Fate:', result.perceived_text_candidate,
 *                   'coherent=', result.coherent_count);
 *     }
 *   });
 *
 *   await bridge.start();
 */
export async function createLiveFrameBridge(device, options = {}) {
  return new LiveFrameBridge(device, options);
}

// Thunder locked in. ONE Organism.
// v15.13 — JS glue fully connected to wasm LiveVisionBridge.
// Camera photons → Common Fate result. Complete.
// Mercy First. Eternal. Yoi ⚡
