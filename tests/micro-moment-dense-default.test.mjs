/**
 * Dense-default micro-moment path (2026-10-10).
 * Local CPU frames only. No network. No incident adjudication.
 * Contact: info@Rathor.ai
 */
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { MercyMotionVisionEngine } from '../mercy-motion-vision-engine.js';
import { buildFailureModeFrames } from '../one-organism-launch.js';
import { makeSyntheticTransferFrames } from '../docs/archive/root-dirs/research/science/s1-micro-moment/harness/system_c_bridge.mjs';

const repo = dirname(dirname(fileURLToPath(import.meta.url)));
const CLAIM = 'Local inspectable research software. Not a certified model upgrade.';

function read(rel) {
  return readFileSync(join(repo, rel), 'utf8');
}

function frame(width, height, timestampMs, paint) {
  const data = new Uint8ClampedArray(width * height);
  paint(data, width, height);
  return { width, height, data, timestampMs };
}

function staticFrames(n = 4, value = 40) {
  const frames = [];
  for (let i = 0; i < n; i++) {
    frames.push(frame(64, 64, i * (1000 / 30), (data) => data.fill(value)));
  }
  return frames;
}

function blurFrames() {
  const frames = [];
  for (let i = 0; i < 4; i++) {
    const t = i * (1000 / 30);
    frames.push(frame(64, 64, t, (data, w, h) => {
      if (i % 2 === 0) {
        for (let y = 0; y < h; y++) {
          for (let x = 0; x < w; x++) data[y * w + x] = x < w / 2 ? 200 : 30;
        }
      } else {
        data.fill(115);
      }
    }));
  }
  return frames;
}

function intervalFrames() {
  const w = 64;
  const h = 64;
  const fps = 30;
  const n = 24;
  const frames = [];
  let step = 0;
  for (let i = 0; i < n; i++) {
    const t = (i / fps) * 1000;
    const inBurst = i >= 9 && i <= 16;
    if (inBurst) step += 1;
    const xOff = inBurst ? step * 10 : 0;
    frames.push(frame(w, h, t, (data) => {
      data.fill(40);
      if (!inBurst) return;
      const bx = 8 + xOff;
      const by = 24;
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          if (x >= bx && x < bx + 14 && y >= by && y < by + 14) data[y * w + x] = 240;
        }
      }
    }));
  }
  return frames;
}

function assertPayload(result) {
  assert.ok(Array.isArray(result.keyMicroMoments), 'keyMicroMoments array');
  assert.ok(Array.isArray(result.causalChain), 'causalChain array');
  assert.equal(result.causalChain.length, result.keyMicroMoments.length);
  assert.equal(result.integration.keyMicroMoments, result.keyMicroMoments);
  assert.equal(result.integration.causalChain, result.causalChain);
  assert.equal(result.integration.claimNote, CLAIM);
  assert.equal(result.sampling.note, CLAIM);
  assert.equal(result.sampling.denseSamplingDefault, true);
  assert.equal(result.opticalFlowMode, 'cpu-dense-fallback');
  result.causalChain.forEach((event, idx) => {
    assert.equal(event.id, `evt_${idx}`);
    assert.equal(typeof event.t, 'number');
    assert.ok(Array.isArray(event.causes));
    assert.ok(Array.isArray(event.effects));
  });
}

const engine = new MercyMotionVisionEngine({ valence: 1.0 });

const quiet = await engine.comprehendVideoStory(staticFrames(), { valence: 1.0 });
assertPayload(quiet);
assert.equal(quiet.sampling.denseSampling, true, 'omitted denseSampling stays on');
assert.equal(quiet.sampling.highMotion, false);
assert.equal(quiet.sampling.motionBlur, false);
assert.equal(quiet.sampling.microBurstWindowMs, 180);
assert.equal(quiet.sampling.targetFpsRequested, 30);
assert.equal(quiet.sampling.resample, 'none');
assert.equal(quiet.keyMicroMoments.length, 0);
assert.equal(quiet.causalChain.length, 0);

const blur = await engine.comprehendVideoStory(blurFrames(), { valence: 1.0 });
assertPayload(blur);
assert.equal(blur.sampling.denseSampling, true);
assert.equal(blur.sampling.motionBlur, true);
assert.equal(blur.sampling.highMotion, true);
assert.equal(blur.sampling.denseApplied, true);
assert.ok(blur.sampling.microBurstWindowMs <= 150, 'blur tightens the micro-burst window');
assert.ok(blur.sampling.targetFpsRequested >= 60, 'blur raises target fps');
assert.equal(blur.sampling.resample, 'cpu-optical-flow-fallback');
assert.ok(blur.keyMicroMoments.length > 0, 'blur still yields local micro-moments');
blur.keyMicroMoments.forEach((m) => assert.ok(m.windowMs <= 150));

const optedOut = await engine.comprehendVideoStory(blurFrames(), {
  valence: 1.0,
  denseSampling: false
});
assertPayload(optedOut);
assert.equal(optedOut.sampling.denseSampling, false);
assert.equal(optedOut.sampling.denseApplied, false);
assert.equal(optedOut.sampling.microBurstWindowMs, 180);
assert.equal(optedOut.sampling.targetFpsRequested, 8);
assert.equal(optedOut.sampling.resample, 'none');
assert.equal(optedOut.opticalFlowMode, 'cpu-dense-fallback');

const interval = await engine.comprehendVideoStory(intervalFrames(), { valence: 1.0 });
assertPayload(interval);
assert.equal(interval.sampling.highSaliency, true);
assert.equal(interval.sampling.denseApplied, true);
assert.equal(interval.sampling.resample, 'cpu-window-focus');
assert.ok(interval.sampling.microBurstWindowMs <= 150);
assert.ok(interval.sampling.targetFpsRequested >= 60);
assert.ok(interval.sampling.focusStartMs > 0, 'focus leaves the quiet lead-in');
assert.ok(interval.sampling.focusEndMs < (23 / 30) * 1000, 'focus leaves the quiet tail');
assert.ok(interval.keyMicroMoments.length > 0);
interval.keyMicroMoments.forEach((m) => {
  assert.ok(m.t >= interval.sampling.focusStartMs - 0.5);
  assert.ok(m.t <= interval.sampling.focusEndMs + 0.5);
});
assert.equal(interval.denseSamplingMode.includes('high-motion-window'), true);

const transfer = await engine.comprehendVideoStory(makeSyntheticTransferFrames(), { valence: 1.0 });
assertPayload(transfer);
assert.equal(transfer.opticalFlowMode, 'cpu-dense-fallback');
assert.ok(transfer.keyMicroMoments.length >= 4, 'local fallback still returns the synthetic transfer bursts');
assert.ok(transfer.keyMicroMoments.some((m) => m.t >= 750 && m.t <= 900));
assert.equal(transfer.sampling.resample === 'cpu-window-focus' || transfer.sampling.resample === 'cpu-optical-flow-fallback', true);

const failure = await engine.analyzeXVideoFailureModes(buildFailureModeFrames(), {
  valence: 1.0,
  denseSampling: true,
  expectedTheft: true,
  expectedRPS: true
});
assert.equal(failure.keyMicroMoments.length, 12);
assert.equal(failure.causalChain.length, 12);
assert.equal(failure.opticalFlowMode, 'cpu-dense-fallback');
assert.equal(failure.sampling.denseSampling, true);
assert.ok(failure.sampling.microBurstWindowMs <= 150);
assert.equal(failure.recoveredDetail, undefined);
assert.equal(failure.story.includes('Classified types: micro_event'), true);
assert.equal(failure.story.includes('object transfers'), false);

const calls = [];
const videoEngine = new MercyMotionVisionEngine({ valence: 1.0 });
videoEngine.extractFramesFromVideoElement = async (_video, options) => {
  calls.push({
    targetFps: options.targetFps,
    startSec: options.startSec,
    endSec: options.endSec
  });
  if (calls.length === 1) return intervalFrames();
  const focused = intervalFrames().filter((f) => f.timestampMs >= 200 && f.timestampMs <= 600);
  return focused.length >= 3 ? focused : intervalFrames().slice(8, 18);
};
const fakeVideo = { currentTime: 0, duration: 0.8, videoWidth: 64, videoHeight: 64, readyState: 4 };
const videoResult = await videoEngine.comprehendVideoStory(fakeVideo, { valence: 1.0, maxDuration: 3 });
assertPayload(videoResult);
assert.equal(calls.length, 2, 'high motion requests a second local seek');
assert.ok(calls[1].targetFps >= 60, 'second seek raises target fps');
assert.equal(typeof calls[1].startSec, 'number');
assert.ok(calls[1].endSec > calls[1].startSec);
assert.ok(calls[1].endSec - calls[1].startSec < 0.8);
assert.equal(videoResult.denseSamplingMode, 'video-element-canvas-high-motion');
assert.equal(videoResult.sampling.resample, 'video-element-canvas');
assert.ok(videoResult.sampling.microBurstWindowMs <= 150);

const page = read('micro-moment.html');
assert.equal(page.includes(CLAIM), true);
assert.equal(page.includes('id="opt-dense" type="checkbox" checked'), true);
assert.equal(page.includes('Dense sampling is the default'), true);
assert.equal(page.includes('targetFps: 24'), false);
assert.equal(page.includes('>keyMicroMoments<'), true);
assert.equal(page.includes('>causalChain<'), true);
assert.equal(page.includes('upgrades Grok'), false);
assert.equal(page.includes('id="mm-claim"'), true);

const doc = read('docs/MICRO_MOMENT_TEMPORAL_COMPREHENSION_v1.0.md');
assert.equal(doc.includes('science/PATSAGI-COUNCIL-MINUTE-2026-10-10-MICRO-MOMENT-DENSE-DEFAULT.md'), true);
assert.equal(doc.includes('science/TASK-CARD-CURSOR-2026-10-10-MICRO-MOMENT-DENSE-DEFAULT.md'), true);

const cargo = read('Cargo.toml');
assert.equal(cargo.includes('version = "14.15.6"'), true);
assert.equal(cargo.includes('crates/mercy-security'), true);

console.log('micro-moment dense default: ok');
