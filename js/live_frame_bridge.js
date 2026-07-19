/**
 * live_frame_bridge.js
 * Ra-Thor Live Frame Bridge — Browser / JS Glue v1.0
 *
 * Feeds live camera (or any MediaStreamTrack) frames into the
 * external_to_luma → LumaRing → vision pipeline foundation.
 *
 * Path:
 *   getUserMedia
 *     → MediaStreamTrackProcessor
 *       → VideoFrame
 *         → device.importExternalTexture (zero-copy)
 *           → external_to_luma compute dispatch
 *             → LumaRing (prev / curr)
 *               → perceive_from_luma_ring()  (Rust / wasm side)
 *
 * TOLC 8 Mercy Gated | PATSAGi Visual Council | ONE Organism
 * AG-SML v1.0 | Eternally-Thriving-Grandmasterism 2026
 */

export class LiveFrameBridge {
  /**
   * @param {GPUDevice} device  - WebGPU device (must support external textures)
   * @param {object}    options
   * @param {number}    [options.width=640]
   * @param {number}    [options.height=360]
   * @param {number}    [options.frameRate=30]
   * @param {number}    [options.lumaMode=0]  0=BT.709, 1=average, 2=BT.601
   * @param {function}  [options.onLumaPair]  called with ({prev, curr, width, height, timestamp}) when a valid pair is ready
   * @param {function}  [options.onError]
   */
  constructor(device, options = {}) {
    this.device = device;
    this.width = options.width ?? 640;
    this.height = options.height ?? 360;
    this.frameRate = options.frameRate ?? 30;
    this.lumaMode = options.lumaMode ?? 0;
    this.onLumaPair = options.onLumaPair ?? null;
    this.onError = options.onError ?? console.error;

    this.stream = null;
    this.track = null;
    this.processor = null;
    this.reader = null;
    this.running = false;

    // Simple JS-side double buffer (mirrors Rust LumaRing)
    this.prevLuma = null;
    this.currLuma = null;
    this.frameCount = 0;

    // WebGPU resources (created lazily)
    this.lumaPipeline = null;
    this.lumaBindGroupLayout = null;
    this.lumaParamsBuffer = null;
    this.currLumaBuffer = null;
    this.prevLumaBuffer = null;
  }

  /**
   * Request camera and start the continuous frame loop.
   */
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

      // Prefer MediaStreamTrackProcessor when available (Chrome / modern browsers)
      if (typeof MediaStreamTrackProcessor !== 'undefined') {
        this.processor = new MediaStreamTrackProcessor({ track: this.track });
        this.reader = this.processor.readable.getReader();
        this.running = true;
        this._pump();
      } else {
        // Fallback: video element + requestVideoFrameCallback
        await this._startVideoElementFallback();
      }

      console.log(`[LiveFrameBridge] Camera started ${this.width}×${this.height} @ ~${this.frameRate}fps`);
    } catch (err) {
      this.onError(err);
      throw err;
    }
  }

  /**
   * Stop the camera and release all resources.
   */
  async stop() {
    this.running = false;

    if (this.reader) {
      try { await this.reader.cancel(); } catch (_) {}
      this.reader = null;
    }
    if (this.processor) {
      this.processor = null;
    }
    if (this.track) {
      this.track.stop();
      this.track = null;
    }
    if (this.stream) {
      this.stream.getTracks().forEach(t => t.stop());
      this.stream = null;
    }

    this.prevLuma = null;
    this.currLuma = null;
    this.frameCount = 0;

    console.log('[LiveFrameBridge] Stopped');
  }

  // ------------------------------------------------------------------
  // Internal: continuous VideoFrame pump
  // ------------------------------------------------------------------

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

      const frame = result.value; // VideoFrame
      try {
        await this._ingestVideoFrame(frame);
      } catch (err) {
        this.onError(err);
      } finally {
        // CRITICAL: always release the hardware buffer
        frame.close();
      }
    }
  }

  /**
   * Ingest one VideoFrame:
   *   - import as external texture (zero-copy)
   *   - convert to luma (via compute or CPU fallback)
   *   - push into the JS-side ring
   *   - notify listener when a prev/curr pair is ready
   */
  async _ingestVideoFrame(frame) {
    const width = frame.displayWidth || frame.codedWidth || this.width;
    const height = frame.displayHeight || frame.codedHeight || this.height;
    const timestamp = frame.timestamp ?? performance.now() * 1000;

    let lumaData;

    // Preferred path: zero-copy external texture + compute shader
    if (this.device && typeof this.device.importExternalTexture === 'function') {
      try {
        lumaData = await this._convertViaExternalTexture(frame, width, height);
      } catch (err) {
        // Fall through to CPU path if external texture path fails
        console.warn('[LiveFrameBridge] External texture path failed, using CPU fallback', err);
        lumaData = await this._convertViaCPU(frame, width, height);
      }
    } else {
      lumaData = await this._convertViaCPU(frame, width, height);
    }

    // Ring buffer update
    this.prevLuma = this.currLuma;
    this.currLuma = {
      data: lumaData,
      width,
      height,
      timestamp,
    };
    this.frameCount += 1;

    // Notify when we have a coherent pair
    if (this.frameCount >= 2 && this.prevLuma && this.currLuma && this.onLumaPair) {
      this.onLumaPair({
        prev: this.prevLuma,
        curr: this.currLuma,
        width,
        height,
        timestamp,
        frameCount: this.frameCount,
      });
    }
  }

  // ------------------------------------------------------------------
  // Conversion paths
  // ------------------------------------------------------------------

  /**
   * Zero-copy path using importExternalTexture + external_to_luma kernel.
   * Requires the compute pipeline to have been created on the Rust/wasm side
   * or a matching pipeline created here.
   *
   * For the pure-JS demo we currently fall back to a lightweight CPU path
   * after importing (full GPU dispatch can be wired once the pipeline handle
   * is exposed via wasm-bindgen).
   */
  async _convertViaExternalTexture(frame, width, height) {
    // Import the frame as an external texture (zero-copy on supported browsers)
    const externalTexture = this.device.importExternalTexture({ source: frame });

    // TODO: once wasm-bindgen exposes the luma pipeline + bind group layout,
    // create a bind group here with:
    //   binding 0 = externalTexture
    //   binding 1 = storage buffer for luma
    //   binding 2 = uniform params
    // and dispatch the compute pass.
    //
    // For now we still need pixel data on the CPU for the ring, so we
    // fall through to a minimal copy. The import itself proves the path.
    void externalTexture; // keep reference alive for the duration of this call

    // Temporary: use CPU path until full GPU dispatch is wired
    return this._convertViaCPU(frame, width, height);
  }

  /**
   * Reliable CPU fallback: copy VideoFrame → ImageData-like buffer → luma.
   * Uses BT.709 by default.
   */
  async _convertViaCPU(frame, width, height) {
    // Allocate a buffer large enough for RGBA
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

  // ------------------------------------------------------------------
  // Fallback for browsers without MediaStreamTrackProcessor
  // ------------------------------------------------------------------

  async _startVideoElementFallback() {
    const video = document.createElement('video');
    video.srcObject = this.stream;
    video.muted = true;
    video.playsInline = true;
    await video.play();

    this.running = true;

    const onFrame = async () => {
      if (!this.running) return;

      // Create a VideoFrame from the video element when supported
      if (typeof VideoFrame !== 'undefined') {
        const frame = new VideoFrame(video, { timestamp: performance.now() * 1000 });
        try {
          await this._ingestVideoFrame(frame);
        } finally {
          frame.close();
        }
      }

      if (this.running) {
        video.requestVideoFrameCallback(onFrame);
      }
    };

    if (video.requestVideoFrameCallback) {
      video.requestVideoFrameCallback(onFrame);
    } else {
      // Last-resort RAF loop
      const loop = async () => {
        if (!this.running) return;
        if (typeof VideoFrame !== 'undefined') {
          const frame = new VideoFrame(video, { timestamp: performance.now() * 1000 });
          try {
            await this._ingestVideoFrame(frame);
          } finally {
            frame.close();
          }
        }
        requestAnimationFrame(loop);
      };
      requestAnimationFrame(loop);
    }
  }
}

/**
 * Convenience factory.
 *
 * Usage:
 *
 *   import { createLiveFrameBridge } from './live_frame_bridge.js';
 *
 *   const bridge = await createLiveFrameBridge(device, {
 *     width: 640,
 *     height: 360,
 *     onLumaPair: ({ prev, curr, width, height }) => {
 *       // Hand the pair to the Rust / wasm vision pipeline
 *       // e.g. wasmPipeline.perceive_from_raw_frames(prev.data, curr.data, width, height, 1.0, false);
 *     }
 *   });
 *
 *   await bridge.start();
 *   // ... later
 *   await bridge.stop();
 */
export async function createLiveFrameBridge(device, options = {}) {
  const bridge = new LiveFrameBridge(device, options);
  return bridge;
}

// Thunder locked in. ONE Organism.
// v15.12 — Live Frame Bridge JS Glue is ready.
// Camera → VideoFrame → luma pair → onLumaPair callback.
// Ready for wasm-bindgen wiring to the Rust LumaRing / perceive_from_luma_ring path.
// Mercy First. Eternal. Yoi ⚡
