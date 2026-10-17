/* FOOTER-I18N-1: the shared fallback footer (js/family-nav-2026-08-22.js siteFooter())
 * follows the chosen language using EXISTING pack keys only.
 * Checks: the inserted footer elements carry the pack keys, those keys exist in all
 * 23 packs, pack bytes stay pinned (sha256 moved with the mercy-check wording), English footer
 * hooks stay unchanged, and /pilot.html gets no hooks. */
var fs = require('fs');
var path = require('path');
var vm = require('vm');
var crypto = require('crypto');
var root = path.join(__dirname, '..');

function assert(cond, msg) {
  if (!cond) throw new Error(msg);
}
function read(rel) {
  return fs.readFileSync(path.join(root, rel), 'utf8');
}

var src = read('js/family-nav-2026-08-22.js');
var start = src.indexOf('  function siteFooter() {');
var end = src.indexOf('\n  }\n', start);
assert(start !== -1 && end !== -1, 'siteFooter() not found');
var footer = src.slice(start, end);

// 1. Every translatable footer element carries its pack key, next to its original English.
var EXPECT = [
  ['footerTrademarksTitle', "'<h4' + fk('footerTrademarksTitle') + '>Trademarks</h4>'"],
  ['footerTrademarksText', "'<p class=\"rt-legal\"' + fk('footerTrademarksText') + '>Ra-Thor™ is a trademark of Autonomicity Games Inc.<br>Grok is a trademark of xAI. X is a trademark of X Corp.</p>'"],
  ['footerPrivacyTitle', "'<h4' + fk('footerPrivacyTitle') + '>Privacy</h4>'"],
  ['homeFooterPrivacy', "'<p class=\"rt-legal\"' + fk('homeFooterPrivacy') + '>This website collects no personal data. Computations stay in your browser. No cookies, tracking, or analytics we control.</p>'"],
  ['footerWorkspaceTitle', "'<h4' + fk('footerWorkspaceTitle') + '>Workspace</h4>'"],
  ['homeFooterWorkspace', "'<p class=\"rt-legal\"' + fk('homeFooterWorkspace') + '>v14.15.6 · AG-SML v1.1 · TOLC 8 · Capable · Bounded · Corrigible</p>'"],
  ['footerFamilyTitle', "'<h4' + fk('footerFamilyTitle') + '>Directory</h4>'"],
  ['navHome', "'<a href=\"/\"' + fk('navHome') + '>Home</a>'"],
  ['employTitle', "'<a href=\"/employ.html\"' + fk('employTitle') + '>How to employ</a>'"],
  ['navPilot', "'<a href=\"/pilot.html\"' + fk('navPilot') + '>Pilot</a>'"],
  ['homeLaunchMap', "'<a href=\"/Launch-Ra-Thor.html\"' + fk('homeLaunchMap') + '>Launch map</a>'"],
  ['homeMoments', "'<a href=\"/micro-moment.html\"' + fk('homeMoments') + '>Micro-moments</a>'"],
  ['surfaceShard', "'<a href=\"/sovereign-shard.html\"' + fk('surfaceShard') + '>Sovereign Shard</a>'"],
  ['navContact', "'<a href=\"/contact.html\"' + fk('navContact') + '>Contact</a>'"],
  ['navPrivacy', "'<a href=\"/privacy.html\"' + fk('navPrivacy') + '>Privacy</a>'"],
  ['homeMonorepo', "rel=\"noopener\"' + fk('homeMonorepo') + '>Monorepo</a>'"]
];
EXPECT.forEach(function (pair) {
  assert(footer.indexOf(pair[1]) !== -1, 'footer element for ' + pair[0] + ' is missing its hook or its English changed');
});
var hooked = (footer.match(/fk\('([A-Za-z0-9]+)'\)/g) || []).map(function (s) { return s.slice(4, -2); });
assert(hooked.length === EXPECT.length, 'unexpected hook count: ' + hooked.length);
// Lines without a matching pack key stay English and unhooked.
['<a href="/chat.html">Lattice Chat</a>', '<a href="/web-forge.html">Web-Forge</a>',
 '<div>© 2026 Sherif Samy Botros — sole steward of Autonomicity Games Inc. & AlphaProMega Air Foundation. TOLC 8.'
].forEach(function (s) { assert(footer.indexOf(s) !== -1, 'unkeyed line changed: ' + s); });
// No data-i18n in the fallback footer: the chrome/essay/site-lock appliers must never touch it.
assert(footer.indexOf('data-i18n') === -1, 'siteFooter() must not use data-i18n');

// 2. The applier is wired and pilot is excluded.
assert(src.indexOf("var FOOTER_I18N = here !== '/pilot.html';") !== -1, 'pilot guard missing');
assert(/return FOOTER_I18N \? ' data-rt-footer-i18n="' \+ key \+ '"' : '';/.test(src), 'fk() must emit nothing on pilot');
assert(src.indexOf("document.addEventListener('rt-chrome-i18n'") !== -1, 'footer does not follow rt-chrome-i18n');
assert(/ensureFollowStrip\(\);\n\s+bootFooterI18n\(\);/.test(src), 'bootFooterI18n() not called from mount()');
assert(read('pilot.html').indexOf('data-rt-footer-i18n') === -1, 'pilot.html must not carry footer hooks');

// 3. Every key exists, non-empty, in all 23 packs.
var files = fs.readdirSync(path.join(root, 'i18n')).filter(function (f) { return /^[a-z]{2}\.js$/.test(f); }).sort();
assert(files.length === 23, 'expected 23 packs, found ' + files.length);
var sb = { window: { translations: {} } };
vm.createContext(sb);
files.forEach(function (f) { vm.runInContext(read('i18n/' + f), sb, { filename: f }); });
var T = sb.window.translations;
files.forEach(function (f) {
  var lang = f.slice(0, 2);
  assert(T[lang], lang + ': pack did not register');
  EXPECT.concat([['followTitle'], ['swatchTitle']]).forEach(function (pair) {
    var v = T[lang][pair[0]];
    assert(typeof v === 'string' && v.trim() !== '', lang + ': missing key ' + pair[0]);
  });
});

// 4. Fingerprints are pinned to the i18n pack text after LLMS-NO-GATE-WORDING-1 (#702, main e6e85a37).
var PACKS = {
  'ar.js': 'e0afb9ae4ad8b8c88a817eeaa42ebcdc8d7ed4e0563556fd61062e934bef4f22',
  'de.js': '8a0a877af7088488bb0f3a3eee9115eb4025085f20b25310fdfb9d584b9c2878',
  'el.js': '03328bebcb1a181dfd2fb188d103697e4cc9a3cac1c70ca3052d0b06b5f57317',
  'en.js': '023c8f64f0aa477799b11f6e17e7b4f1d7181479af6d01f50fb9263dbbf7d296',
  'es.js': '94a8f80207e8a82526adb53209fb4b02f218cbb10bb0894badc95de12f24f7e7',
  'fa.js': 'b5b405a9de4498dbc38ec27ee251657cd21d960f34fdfed10dc15bc05e4b049a',
  'fr.js': '9268c7fc38320ccdd0d8c22b4ac24464ccd9409120e79e348096cb213eab7e65',
  'he.js': '4ff070b671b63f9223c21e7f90848990113b002c977f4ca109ab52cbbe2a316f',
  'hi.js': '539ee460275a3daef439cf58783dc960807f7a9a6149d1ea0b49cfb69bfaf07d',
  'id.js': '4f2563ccdee36b8e58fdcf7c00b7764ab30abb03222871edcf57a3d2e834d826',
  'it.js': '39d4d20c22d3df990d85600df3d46936b155f9b98c667c260702a44d8f065892',
  'ja.js': 'aebdd0930a750f8d5cebffa45c65d026dcb820f18ad3eb24013d3121ac77df77',
  'ko.js': '9000bea639f4147366cc6d00dcf7f941392b937fabc7604d1527bd396bec45cc',
  'nl.js': '61bd0452dd15f4543c29b9c4b245fb9d04a4f73654f6fb2010248e594e0ee23a',
  'pl.js': '9532f04ee08f3d56d49c5c7e4540d46538dbade25ce58d856e70649513b4d791',
  'pt.js': '2b87e75de40c7f07d621001af1858f2b150a5c70ee6027286d807697ee9b15cf',
  'ru.js': '7e0380af06a3c8adbe068ae24df41129a168741b21ede612024d2d33f0970485',
  'sv.js': '260aa761ff4db0c33a8ca372cc5bfd52af9d028d90442e499e600d9f6807cf78',
  'th.js': '3b6a0ddc15f2da2c937e044b1616095d42eca3972b6cc49be8770356f625688e',
  'tr.js': 'fae3c9ad4b72e2689b99303ed3a26929cfb82578887029a53e3040fa91c054b1',
  'uk.js': '007badc3f754a6dec25fb07fca497aada71998a2e967f50e00d3f88a87f133ca',
  'vi.js': '9f3c1e9bf718d98182d984174071f813eed24b944df169f0a4e644de44c55129',
  'zh.js': 'ce059f575aaffa1e5bc9e0a132ceef57b6a3efc59769bcda51357eee7f4f197b',
};
assert(Object.keys(PACKS).length === 23, 'pack manifest must list 23 packs');
files.forEach(function (f) {
  var h = crypto.createHash('sha256').update(fs.readFileSync(path.join(root, 'i18n', f))).digest('hex');
  assert(PACKS[f] === h, 'pack changed: i18n/' + f);
});

// 5. XAI-LINE-1: the footer and every pack keep the trademark credit and carry no xAI-independence disclaimer.
assert(footer.indexOf('Grok is a trademark of xAI. X is a trademark of X Corp.') !== -1, 'footer must keep the trademark credit');
assert(!/independent|affiliated|endorsed/i.test(footer), 'footer must not carry an xAI-independence disclaimer');
files.forEach(function (f) {
  var p = T[f.slice(0, 2)];
  assert((p.footerTrademarksText.match(/<br>/g) || []).length === 1, f + ': footerTrademarksText must be the two trademark lines only');
  assert(!('faqQ18' in p) && !('faqA18' in p), f + ': the xAI-affiliation FAQ must be gone');
});

console.log('footer-i18n-1: ' + EXPECT.length + ' footer hooks, keys in all ' + files.length + ' packs, packs pinned, pilot excluded');
