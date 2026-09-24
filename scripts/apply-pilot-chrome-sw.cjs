#!/usr/bin/env node
/**
 * Idempotent PILOT-CHROME-SW apply for branch grokbot/pilot-chrome-sw-20260924
 * Contact: info@Rathor.ai · workspace 14.15.6
 */
const fs = require('fs');
const path = require('path');
const root = process.cwd();
const OLD = '20260923f';
const NEW = '20260924a';
const navPilot = {
  en: 'Pilot', ar: 'تجريبي', es: 'Piloto', fr: 'Pilote', nl: 'Proef',
  de: 'Pilot', zh: '试点', ja: 'パイロット', pt: 'Piloto', ru: 'Пилот',
  hi: 'पायलट', it: 'Pilota', ko: '파일럿', uk: 'Пілот', pl: 'Pilot',
  tr: 'Pilot', vi: 'Thử nghiệm', id: 'Uji coba', sv: 'Pilot', th: 'ไพลอต',
  el: 'Πιλότος', fa: 'پایلوت', he: 'פיילוט'
};
function read(rel) { return fs.readFileSync(path.join(root, rel), 'utf8'); }
function write(rel, t) { fs.writeFileSync(path.join(root, rel), t); console.log('wrote', rel); }

// 1 chrome
let chrome = read('js/i18n-chrome.js');
if (!chrome.includes('navPilot: 1')) {
  chrome = chrome.replace('navHome: 1, navChat: 1, navEmploy: 1, navLaunch: 1,',
    'navHome: 1, navChat: 1, navEmploy: 1, navPilot: 1, navLaunch: 1,');
}
if (!chrome.includes("'/pilot.html': 'navPilot'")) {
  chrome = chrome.replace("'/employ.html': 'navEmploy',\n",
    "'/employ.html': 'navEmploy',\n    '/pilot.html': 'navPilot',\n");
}
chrome = chrome.replace(`var PACK_V = '${OLD}';`, `var PACK_V = '${NEW}';`);
chrome = chrome.replace(`var PACK_V = '${NEW}';`, `var PACK_V = '${NEW}';`);
// ensure semicolon after RTL_RE
chrome = chrome.replace(
  /var RTL_RE = \/\[\\u0590-\\u08FF\\uFB1D-\\uFDFF\\uFE70-\\uFEFF\]\/(?!;)/,
  "var RTL_RE = /[\\u0590-\\u08FF\\uFB1D-\\uFDFF\\uFE70-\\uFEFF]/;"
);
// also plain form
if (!/var RTL_RE = \/\[[^\]]+\]\/;/.test(chrome)) {
  chrome = chrome.replace(/var RTL_RE = (\/\[[^\]]+\]\/)\s*\n/, 'var RTL_RE = $1;\n');
}
write('js/i18n-chrome.js', chrome);

// 2 packs
for (const [lang, val] of Object.entries(navPilot)) {
  const rel = `i18n/${lang}.js`;
  let t = read(rel);
  if (!t.includes('"navPilot"')) {
    const idx = t.indexOf('"navEmploy":');
    if (idx < 0) throw new Error('missing navEmploy in ' + lang);
    const lineEnd = t.indexOf('\n', idx);
    const insert = `\n  "navPilot": ${JSON.stringify(val)},`;
    t = t.slice(0, lineEnd) + insert + t.slice(lineEnd);
    write(rel, t);
  } else console.log('skip', rel);
}

// 3 sw
let sw = read('sw.js');
sw = sw.split(OLD).join(NEW);
if (!sw.includes("'/pilot.html'")) {
  sw = sw.replace("'/employ.html',", "'/employ.html', '/pilot.html',");
}
write('sw.js', sw);

// 4 lock pins
const lockFiles = [
  'js/site-lock-2026-08-22.js', 'i18n/README.md', 'js/i18n-essay.js',
  'index.html','employ.html','privacy.html','contact.html','chat.html',
  'go-x.html','Launch-Ra-Thor.html','constellation-week.html','offline.html',
  'thanks.html','briefing.html','science-watches.html','pilot.html',
  'tests/chrome-pack-keys.test.js','tests/recent-updates-i18n.test.js'
];
for (const rel of lockFiles) {
  let t = read(rel);
  if (t.includes(OLD)) { write(rel, t.split(OLD).join(NEW)); }
  else console.log('lock ok', rel);
}

// 5 test asserts
let test = read('tests/chrome-pack-keys.test.js');
if (!test.includes("KEYS.indexOf('navPilot')")) {
  test = test.replace(
    "assert(KEYS.indexOf('navEmploy') !== -1, 'navEmploy must stay applied');\n",
    "assert(KEYS.indexOf('navEmploy') !== -1, 'navEmploy must stay applied');\nassert(KEYS.indexOf('navPilot') !== -1, 'navPilot must stay applied');\n"
  );
}
if (!test.includes("RTL_RE.test(pack.navPilot)")) {
  test = test.replace(
    "assert(RTL_RE.test(pack.navEmploy), lang + ' navEmploy must be RTL script');\n",
    "assert(RTL_RE.test(pack.navEmploy), lang + ' navEmploy must be RTL script');\n    assert(RTL_RE.test(pack.navPilot), lang + ' navPilot must be RTL script');\n"
  );
}
if (!test.includes('sw precache must list /pilot.html')) {
  test = test.replace(
    "assert(sw.indexOf(\"'/js/i18n-chrome.js'\") !== -1, 'sw precache must list i18n-chrome.js');\n",
    "assert(sw.indexOf(\"'/js/i18n-chrome.js'\") !== -1, 'sw precache must list i18n-chrome.js');\nassert(sw.indexOf(\"'/pilot.html'\") !== -1, 'sw precache must list /pilot.html');\n"
  );
}
if (!test.includes("'pilot.html'")) {
  test = test.replace(
    "'thanks.html', 'briefing.html', 'science-watches.html'\n];",
    "'thanks.html', 'briefing.html', 'science-watches.html', 'pilot.html'\n];"
  );
}
write('tests/chrome-pack-keys.test.js', test);
console.log('apply-pilot-chrome-sw: done');
