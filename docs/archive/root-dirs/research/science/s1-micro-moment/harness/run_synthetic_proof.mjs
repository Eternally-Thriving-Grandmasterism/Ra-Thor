/**
 * S-1 Rank 2 — synthetic end-to-end proof
 * Runs metrics harness against fixtures; optional engine path documented.
 * Contact: info@Rathor.ai
 */
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { evaluateDataset } from './metrics.mjs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const labels = JSON.parse(
  readFileSync(join(__dirname, '../fixtures/synthetic_labels.json'), 'utf8')
);

// System A (sparse): often misses short events — simulated miss on E1
const systemA = {
  syn_e1_001: [],
  syn_e4_001: [],
};

// System C (Ra-Thor-class): recovered micro-burst window
const systemC = {
  syn_e1_001: [{ t_start_ms: 790, t_end_ms: 1120, class: 'E1_object_transfer' }],
  syn_e4_001: [],
};

const reportA = evaluateDataset(labels.clips, systemA, 0.5);
const reportC = evaluateDataset(labels.clips, systemC, 0.5);

const out = {
  mission: 'S-1 Rank 2',
  note: 'Synthetic proof of harness + claim-gate shape — NOT a scientific result on real video',
  system_A_sparse_sim: { recall: reportA.recall, precision: reportA.precision },
  system_C_rathor_sim: { recall: reportC.recall, precision: reportC.precision },
  claim_gate_shape:
    'On real data: C must beat A on recall for E1–E3 with reported CI',
  contact: 'info@Rathor.ai',
};

console.log(JSON.stringify(out, null, 2));

if (!(reportC.recall > reportA.recall)) {
  console.error('Synthetic ordering failed');
  process.exit(1);
}
console.error('S-1 synthetic proof: OK (C recall > A recall on fixtures)');
