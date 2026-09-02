/**
 * S-1 Micro-Moment Benchmark — metrics harness (pure JS, zero deps)
 * Ra-Thor · PATSAGi · Science Mission S-1 Rank 2
 * Contact: info@Rathor.ai
 *
 * Evaluates predicted events vs gold labels (span IoU matching).
 * Does not invent scientific conclusions — prints numbers only.
 */

export function spanIoU(a, b) {
  const start = Math.max(a.t_start_ms, b.t_start_ms);
  const end = Math.min(a.t_end_ms, b.t_end_ms);
  const inter = Math.max(0, end - start);
  const union =
    Math.max(a.t_end_ms, b.t_end_ms) - Math.min(a.t_start_ms, b.t_start_ms);
  return union <= 0 ? 0 : inter / union;
}

/**
 * Match predictions to gold events (greedy by IoU).
 * @param {Array} goldEvents
 * @param {Array} predEvents  each { t_start_ms, t_end_ms, class? }
 * @param {number} iouThr
 */
export function matchEvents(goldEvents, predEvents, iouThr = 0.5) {
  const gold = goldEvents.filter((e) => e.class !== 'E4_negative');
  const preds = [...predEvents];
  const matched = [];
  const usedPred = new Set();

  for (const g of gold) {
    let best = null;
    let bestIou = 0;
    for (let i = 0; i < preds.length; i++) {
      if (usedPred.has(i)) continue;
      const iou = spanIoU(g, preds[i]);
      if (iou >= iouThr && iou > bestIou) {
        bestIou = iou;
        best = i;
      }
    }
    if (best != null) {
      usedPred.add(best);
      matched.push({
        gold: g,
        pred: preds[best],
        iou: bestIou,
        timing_error_ms: Math.abs(
          (preds[best].t_start_ms + preds[best].t_end_ms) / 2 -
            (g.t_start_ms + g.t_end_ms) / 2
        ),
      });
    }
  }

  const fp = preds.length - usedPred.size;
  const fn = gold.length - matched.length;
  const tp = matched.length;
  const recall = gold.length === 0 ? 1 : tp / gold.length;
  const precision = preds.length === 0 ? (gold.length === 0 ? 1 : 0) : tp / preds.length;

  return {
    tp,
    fp,
    fn,
    recall,
    precision,
    median_timing_error_ms: median(matched.map((m) => m.timing_error_ms)),
    matches: matched,
  };
}

function median(xs) {
  if (!xs.length) return null;
  const s = [...xs].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

/**
 * Evaluate a full dataset of clips.
 * predictions: { [clip_id]: predEvent[] }
 */
export function evaluateDataset(clips, predictions, iouThr = 0.5) {
  const perClip = [];
  let tp = 0,
    fp = 0,
    fn = 0;
  const timing = [];

  for (const clip of clips) {
    const preds = predictions[clip.clip_id] || [];
    const r = matchEvents(clip.events || [], preds, iouThr);
    perClip.push({ clip_id: clip.clip_id, ...r });
    tp += r.tp;
    fp += r.fp;
    fn += r.fn;
    for (const m of r.matches) timing.push(m.timing_error_ms);
  }

  const goldPos = tp + fn;
  return {
    iou_threshold: iouThr,
    tp,
    fp,
    fn,
    recall: goldPos === 0 ? 1 : tp / goldPos,
    precision: tp + fp === 0 ? 1 : tp / (tp + fp),
    median_timing_error_ms: median(timing),
    per_clip: perClip,
    contact: 'info@Rathor.ai',
  };
}

// CLI smoke when run directly
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));

if (process.argv[1] && process.argv[1].endsWith('metrics.mjs')) {
  const labels = JSON.parse(
    readFileSync(join(__dirname, '../fixtures/synthetic_labels.json'), 'utf8')
  );
  // Synthetic predictions: hit E1 roughly, stay quiet on negative
  const predictions = {
    syn_e1_001: [{ t_start_ms: 820, t_end_ms: 1080, class: 'E1_object_transfer' }],
    syn_e4_001: [],
  };
  const report = evaluateDataset(labels.clips, predictions, 0.5);
  console.log(JSON.stringify(report, null, 2));
  if (report.recall < 1 || report.fp > 0) {
    console.error('Synthetic claim-gate unexpected — check fixtures');
    process.exit(1);
  }
  console.error('S-1 metrics harness smoke: OK');
}
