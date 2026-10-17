#!/usr/bin/env node
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.join(__dirname, '..');
const patches = JSON.parse(fs.readFileSync(path.join(__dirname, 'licence-v12-i18n-patches.json'), 'utf8'));

function setKey(src, key, value) {
  const esc = value.replace(/\\/g, '\\\\').replace(/"/g, '\\"');
  const re = new RegExp('("' + key + '": ")(?:[^"\\\\]|\\\\.)*(")', 'g');
  if (!re.test(src)) throw new Error('missing key ' + key);
  return src.replace(re, '$1' + esc + '$2');
}

const langs = Object.keys(patches).sort();
for (const lang of langs) {
  const file = path.join(root, 'i18n', lang + '.js');
  let src = fs.readFileSync(file, 'utf8');
  const patch = patches[lang];
  for (const key of Object.keys(patch)) {
    src = setKey(src, key, patch[key]);
  }
  fs.writeFileSync(file, src);
  console.log('patched i18n/' + lang + '.js');
}
