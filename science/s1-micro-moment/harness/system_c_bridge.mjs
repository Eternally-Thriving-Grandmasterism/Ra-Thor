/**
 * S-1 System C bridge — MercyMotionVisionEngine → prediction events → metrics
 * Ra-Thor · PATSAGi · Contact: info@Rathor.ai
 */

import { readFileSync } from 'fs';
import { fileURLToPath, pathToFileURL } from 'url';
import { dirname, join } from 'path';
import { evaluateDataset } from './metrics.mjs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, '../../..');

const TYPE_TO_CLASS = {
  object_transfer: 'E1_object_transfer',
  gesture_sequence: 'E2_gesture_resolution',
  contact: 'E3_contact_onset',
  micro_event: 'E3_contact_onset',
};

/** Merge consecutive bursts of same mapped class within gapMs into spans. */
export function burstsToPredictions(keyMicroMoments, options = {}) {
  const gapMs = options.gapMs ?? 120;
  const halfWindow = options.halfWindowMs ?? 80;
  if (!keyMicroMoments?.length) return [];

  const sorted = [...keyMicroMoments].sort((a, b) => a.t - b.t);
  const spans = [];
  let cur = null;

  for (const b of sorted) {
    const cls = TYPE_TO_CLASS[b.type] || 'E3_contact_onset';
    const t0 = Math.max(0, b.t - halfWindow);
    const t1 = b.t + halfWindow;
    if (!cur || cur.class !== cls || t0 - cur.t_end_ms > gapMs) {
      if (cur) spans.push(cur);
      cur = { t_start_ms: t0, t_end_ms: t1, class: cls };
    } else {
      cur.t_end_ms = Math.max(cur.t_end_ms, t1);
    }
  }
  if (cur) spans.push(cur);
  return spans;
}

/**
 * Synthetic transfer: bright block jumps ~10px per frame during burst window
 * so centroid dx exceeds engine object_transfer speed threshold (~4.5).
 */
export function makeSyntheticTransferFrames({
  width = 64,
  height = 64,
  fps = 30,
  durationMs = 2000,
  burstStartMs = 800,
  burstEndMs = 1100,
} = {}) {
  const n = Math.max(3, Math.round((durationMs / 1000) * fps));
  const frames = [];
  let step = 0;
  for (let i = 0; i < n; i++) {
    const t = (i / fps) * 1000;
    const data = new Uint8ClampedArray(width * height);
    const inBurst = t >= burstStartMs && t <= burstEndMs;
    if (inBurst) step += 1;
    // Large per-frame translation during burst → high dx in block matching
    const xOff = inBurst ? step * 10 : 0;
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        let v = 40;
        if (inBurst) {
          const bx = 8 + xOff;
          const by = 24;
          if (x >= bx && x < bx + 14 && y >= by && y < by + 14) v = 240;
        }
        data[y * width + x] = v;
      }
    }
    frames.push({ width, height, data, timestampMs: t });
  }
  frames._extractionMode = 'synthetic-s1-bridge';
  return frames;
}

export async function runSystemCOnFrames(frames, engineOptions = {}) {
  const enginePath = pathToFileURL(join(ROOT, 'mercy-motion-vision-engine.js')).href;
  const mod = await import(enginePath);
  const Engine = mod.MercyMotionVisionEngine || mod.default?.MercyMotionVisionEngine;
  if (!Engine) throw new Error('MercyMotionVisionEngine not found');
  const engine = new Engine({ debugMode: false, valence: 1.0 });
  const result = await engine.comprehendVideoStory(frames, {
    query: 's1 system c micro-moment recovery',
    valence: 1.0,
    ...engineOptions,
  });
  if (result.error) throw new Error(result.error);
  const predictions = burstsToPredictions(result.keyMicroMoments || []);
  return { result, predictions };
}

if (process.argv[1] && process.argv[1].endsWith('system_c_bridge.mjs')) {
  const labels = JSON.parse(
    readFileSync(join(__dirname, '../fixtures/synthetic_labels.json'), 'utf8')
  );
  const frames = makeSyntheticTransferFrames();
  const { result, predictions } = await runSystemCOnFrames(frames);
  const types = [...new Set((result.keyMicroMoments || []).map((m) => m.type))];
  const predMap = {
    syn_e1_001: predictions,
    syn_e4_001: [],
  };
  const report = evaluateDataset(labels.clips, predMap, 0.5);
  const out = {
    mission: 'S-1 System C bridge',
    note: 'Synthetic pixel proof — NOT real-video science',
    engine: result.engine,
    burst_types: types,
    engine_bursts: (result.keyMicroMoments || []).length,
    predictions_e1: predictions,
    metrics: {
      recall: report.recall,
      precision: report.precision,
      tp: report.tp,
      fp: report.fp,
      fn: report.fn,
    },
    contact: 'info@Rathor.ai',
  };
  console.log(JSON.stringify(out, null, 2));
  if ((result.keyMicroMoments || []).length === 0) {
    console.error('System C bridge: no bursts');
    process.exit(1);
  }
  if (report.recall < 1) {
    console.error('System C bridge: expected synthetic E1 span match');
    process.exit(1);
  }
  console.error('S-1 System C bridge: OK');
}
