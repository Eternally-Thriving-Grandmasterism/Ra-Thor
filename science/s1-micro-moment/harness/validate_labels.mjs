/**
 * S-1 label structural validator
 * Usage: node validate_labels.mjs <file.json|directory>
 * Contact: info@Rathor.ai
 */
import { readFileSync, readdirSync, statSync } from 'fs';
import { join } from 'path';

const CLASSES = new Set([
  'E1_object_transfer',
  'E2_gesture_resolution',
  'E3_contact_onset',
  'E4_negative',
]);
const SPLITS = new Set(['train', 'val', 'test']);
const SOURCE_TYPES = new Set(['file', 'url', 'synthetic']);

function validateClip(c, name) {
  const errs = [];
  if (!c || typeof c !== 'object') return [`${name}: not an object`];
  if (!c.clip_id) errs.push(`${name}: missing clip_id`);
  if (!c.source || !SOURCE_TYPES.has(c.source.type))
    errs.push(`${name}: source.type must be file|url|synthetic`);
  if (typeof c.duration_ms !== 'number' || c.duration_ms < 0)
    errs.push(`${name}: duration_ms`);
  if (typeof c.fps !== 'number' || c.fps < 1) errs.push(`${name}: fps`);
  if (!SPLITS.has(c.split)) errs.push(`${name}: split`);
  if (!Array.isArray(c.events)) errs.push(`${name}: events array required`);
  else {
    c.events.forEach((e, i) => {
      const p = `${name}.events[${i}]`;
      if (!e.event_id) errs.push(`${p}: event_id`);
      if (!CLASSES.has(e.class)) errs.push(`${p}: class`);
      if (typeof e.t_start_ms !== 'number' || typeof e.t_end_ms !== 'number')
        errs.push(`${p}: times`);
      else if (e.t_end_ms < e.t_start_ms) errs.push(`${p}: end < start`);
    });
  }
  return errs;
}

function loadTargets(arg) {
  const st = statSync(arg);
  if (st.isDirectory()) {
    return readdirSync(arg)
      .filter((f) => f.endsWith('.json') && !f.startsWith('_'))
      .map((f) => join(arg, f));
  }
  return [arg];
}

const arg = process.argv[2];
if (!arg) {
  console.error('Usage: node validate_labels.mjs <file.json|dir>');
  process.exit(2);
}

let allErrs = [];
let n = 0;
for (const path of loadTargets(arg)) {
  const raw = JSON.parse(readFileSync(path, 'utf8'));
  const clips = Array.isArray(raw.clips) ? raw.clips : [raw];
  for (const c of clips) {
    n++;
    allErrs = allErrs.concat(validateClip(c, c.clip_id || path));
  }
}

if (allErrs.length) {
  console.error(JSON.stringify({ ok: false, clips_checked: n, errors: allErrs }, null, 2));
  process.exit(1);
}
console.log(JSON.stringify({ ok: true, clips_checked: n, contact: 'info@Rathor.ai' }, null, 2));
