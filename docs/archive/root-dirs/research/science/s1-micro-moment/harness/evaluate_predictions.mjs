/**
 * Evaluate a predictions JSON map against a labels file.
 * Usage:
 *   node evaluate_predictions.mjs <labels.json> <predictions.json>
 * predictions.json shape: { "clip_id": [ { t_start_ms, t_end_ms, class? }, ... ], ... }
 * Contact: info@Rathor.ai
 */
import { readFileSync } from 'fs';
import { evaluateDataset } from './metrics.mjs';

const labelsPath = process.argv[2];
const predsPath = process.argv[3];
if (!labelsPath || !predsPath) {
  console.error('Usage: node evaluate_predictions.mjs <labels.json> <predictions.json>');
  process.exit(2);
}

const labelsRaw = JSON.parse(readFileSync(labelsPath, 'utf8'));
const clips = Array.isArray(labelsRaw.clips) ? labelsRaw.clips : [labelsRaw];
const predictions = JSON.parse(readFileSync(predsPath, 'utf8'));
const report = evaluateDataset(clips, predictions, 0.5);
console.log(JSON.stringify({ ...report, contact: 'info@Rathor.ai' }, null, 2));
