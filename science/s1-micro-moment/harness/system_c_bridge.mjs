/**
 * S-1 System C bridge — MercyMotionVisionEngine → prediction events → metrics
 * Ra-Thor · PATSAGi · maximal-leverage pure path
 * Contact: info@Rathor.ai
 *
 * Maps engine micro-bursts to { t_start_ms, t_end_ms, class } for evaluateDataset.
 * Synthetic pixel path proves end-to-end without claiming real-video science.
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

/** Build gray frames: quiet then a translating bright block (object_transfer cue). */
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
  for (let i = 0; i < n; i++) {
    const t = (i / fps) * 1000;
    const data = new Uint8ClampedArray(width * height);
    const inBurst = t >= burstStartMs && t <= burstEndMs;
    const xOff = inBurst ? Math.floor(((t - burstStartMs) / Math.max(1, burstEndMs - burstStartMs)) * 20) : 0;
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        let v = 40;
        if (inBurst) {
          const bx = 18 + xOff;
          const by = 24;
          if (x >= bx && x < bx + 12 && y >= by && y < by + 12) v = 220;
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
  const engine = new Engine({ debugMode: false });
  const result = await engine.comprehendVideoStory(frames, {
    query: 's1 system c micro-moment recovery',
    ...engineOptions,
  });
  if (result.error) throw new Error(result.error);
  const predictions = burstsToPredictions(result.keyMicroMoments || []);
  return { result, predictions };
}

// CLI: synthetic end-to-end proof
if (process.argv[1] && process.argv[1].endsWith('system_c_bridge.mjs')) {
  const labels = JSON.parse(
    readFileSync(join(__dirname, '../fixtures/synthetic_labels.json'), 'utf8')
  );
  const frames = makeSyntheticTransferFrames();
  const { result, predictions } = await runSystemCOnFrames(frames);
  const predMap = {
    syn_e1_001: predictions,
    syn_e4_001: [], // negative: no frames / no run
  };
  const report = evaluateDataset(labels.clips, predMap, 0.5);
  const out = {
    mission: 'S-1 System C bridge',
    note: 'Synthetic pixel proof of engine→predictions→metrics — NOT real-video science',
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
  // Soft gate on synthetic: at least one prediction path exercised
  if ((result.keyMicroMoments || []).length === 0 && predictions.length === 0) {
    console.error('System C bridge: no bursts on synthetic transfer — check engine thresholds');
    process.exit(1);
  }
  console.error('S-1 System C bridge: OK');
}
