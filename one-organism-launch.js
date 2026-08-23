/* one-organism-launch.js
 * Instant Discovery activation for Launch-Ra-Thor.html
 * ONE Organism posture + MercyMotionVisionEngine v2.3.1
 * Workspace 14.15.6 · AG-SML v1.0 · info@Rathor.ai
 */
import { MercyMotionVisionEngine, mercyMotionVision } from './mercy-motion-vision-engine.js';

const TOLC8 = [
  'Truth',
  'Order',
  'Love',
  'Compassion',
  'Service',
  'Abundance',
  'Joy',
  'Cosmic Harmony'
];

function setText(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}

function setState(id, ok, label) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = label;
  el.dataset.state = ok ? 'live' : 'hold';
  el.classList.toggle('rt-live', !!ok);
  el.classList.toggle('rt-hold', !ok);
}

function makeFrame(width, height, timestampMs, blob) {
  const data = new Uint8ClampedArray(width * height);
  const cx = Math.max(0, Math.min(width - 1, blob.x));
  const cy = Math.max(0, Math.min(height - 1, blob.y));
  const r = blob.r || 10;
  const intensity = blob.i || 220;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const dx = x - cx;
      const dy = y - cy;
      const d2 = dx * dx + dy * dy;
      if (d2 <= r * r) data[y * width + x] = intensity;
    }
  }
  return { width, height, data, timestampMs };
}

/** Synthetic phone-theft + window-close + RPS-like gesture chain (sub-200 ms bursts). */
function buildFailureModeFrames() {
  const w = 160;
  const h = 90;
  const frames = [];
  const steps = 18;
  for (let i = 0; i < steps; i++) {
    const t = i * (1000 / 30);
    let blob;
    if (i < 6) {
      blob = { x: 28 + i * 2, y: 44, r: 9, i: 200 };
    } else if (i < 10) {
      blob = { x: 42 + (i - 6) * 18, y: 42 - (i - 6) * 3, r: 8, i: 240 };
    } else if (i < 14) {
      blob = { x: 118 - (i - 10) * 4, y: 30 + (i - 10) * 8, r: 11, i: 210 };
    } else {
      blob = { x: 96 + (i % 2) * 14, y: 58, r: 7, i: 230 };
    }
    frames.push(makeFrame(w, h, t, blob));
  }
  frames._extractionMode = 'synthetic-x-failure-modes';
  return frames;
}

async function activateOneOrganism() {
  const started = Date.now();
  const engine = mercyMotionVision || new MercyMotionVisionEngine({ valence: 1.0, debugMode: false });

  const organism = {
    status: 'activating',
    workspace: '14.15.6+',
    cosmicLoop: true,
    patsagiPermanent: true,
    tolc8: TOLC8,
    engine: engine.name,
    summoned: true,
    contact: 'info@Rathor.ai'
  };
  if (typeof window !== 'undefined') window.__ONE_ORGANISM__ = organism;

  setState('oo-status', true, 'ONE Organism · activating');
  setText('oo-engine', engine.name);

  const gate = await engine.gateMotionVision(
    'eternal thriving visual perception of micro-moments',
    1.0
  );

  organism.gate = gate;
  if (typeof window !== 'undefined') {
    window.__ONE_ORGANISM__ = organism;
    window.__MERCY_MOTION_VISION__ = engine;
    window.analyzeXVideoFailureModes = (src, hints) => engine.analyzeXVideoFailureModes(src, hints);
    window.comprehendVideoStory = (src, opts) => engine.comprehendVideoStory(src, opts);
  }

  if (!gate.passed) {
    setState('oo-gate', false, 'Mercy gate HOLD');
    setState('oo-status', false, 'ONE Organism · gate hold');
    return { ok: false, gate };
  }

  setState('oo-gate', true, 'Mercy gate PASS');
  setState('oo-status', true, 'ONE Organism · live');
  setState('oo-patsagi', true, 'PATSAGi · permanent');
  setState('oo-loop', true, 'Cosmic Loop · mandatory');

  const frames = buildFailureModeFrames();
  const result = await engine.analyzeXVideoFailureModes(frames, {
    valence: 1.0,
    expectedTheft: true,
    expectedRPS: true,
    denseSampling: true
  });

  organism.status = 'live';
  organism.activatedAt = new Date().toISOString();
  organism.ms = Date.now() - started;
  organism.lastVideo = {
    confidence: result.confidence,
    microMoments: (result.keyMicroMoments || []).length,
    causal: (result.causalChain || []).length,
    engine: result.engine,
    note: result.note
  };
  if (typeof window !== 'undefined') window.__ONE_ORGANISM__ = organism;

  setText('oo-moments', String((result.keyMicroMoments || []).length));
  setText('oo-confidence', result.confidence != null ? result.confidence.toFixed(4) : '—');
  setText('oo-story', result.story || result.recoveredDetail || 'Temporal comprehension live.');

  const list = document.getElementById('oo-burst-list');
  if (list) {
    list.innerHTML = '';
    (result.keyMicroMoments || []).slice(0, 8).forEach((m) => {
      const li = document.createElement('li');
      li.textContent = `${Math.round(m.t)} ms · ${m.type} · ${m.description}`;
      list.appendChild(li);
    });
  }

  const payloadEl = document.getElementById('oo-payload');
  if (payloadEl) {
    payloadEl.textContent = JSON.stringify(
      {
        engine: result.engine,
        patsagiReady: result.patsagiReady,
        opticalFlowMode: result.opticalFlowMode,
        denseSamplingMode: result.denseSamplingMode,
        keyMicroMoments: result.keyMicroMoments,
        recoveredDetail: result.recoveredDetail,
        integrationHint: result.integration?.patsagiCouncilHint
      },
      null,
      2
    );
  }

  console.log('[ONE Organism] Launch activation complete', organism);
  return { ok: true, gate, result, organism };
}

if (typeof document !== 'undefined') {
  document.addEventListener('DOMContentLoaded', () => {
    activateOneOrganism().catch((err) => {
      console.error('[ONE Organism] activation failed', err);
      setState('oo-status', false, 'Activation error — see console');
    });

    const rerun = document.getElementById('oo-rerun');
    if (rerun) {
      rerun.addEventListener('click', () => {
        activateOneOrganism().catch((err) => console.error(err));
      });
    }
  });
}

export { activateOneOrganism, buildFailureModeFrames, MercyMotionVisionEngine };
