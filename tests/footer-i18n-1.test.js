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
  'ar.js': '14e03bfba238a475fd28d8458781d26ad975a53b2720e45734d30101aa172546',
  'de.js': '4a26c79219ddaffe6ba50910a35f2cfdbb057ce790fdbf3ada7ec85e40f434bd',
  'el.js': '929142a80ecb4047684dc074b399b26c97a0720d09260196abf408456e594027',
  'en.js': 'cfee675ba755edc22952ab6b6a80e54dec2fa7807831800a58e2d1e328578f63',
  'es.js': '37336e0ab0941fb9531aa6a52564d1a4a1e5d1a97ca7661c0ad62d0d41ca75f1',
  'fa.js': '69c4f21b0fa1e916ce4b8895522d7e594dae10d1ecc24908286fdac4a4b34690',
  'fr.js': '6faf3dccd345ed644eb496c3b890df151f954a5f3e703ffce7ef2a5f40adc5ca',
  'he.js': '0d0720afc4ecf59e90982eed0deafd7d397bb82f4513c1c15347ed58302e1421',
  'hi.js': 'bcba98307355ce2ca390c19117fac2e9bc7d41ddd733cda3f4a0158647cc811b',
  'id.js': 'd9c3f64968e9cde05ab001c235163fd0ae08b3bd0a110140b3d259981cbbd664',
  'it.js': 'b8af2643f4ef8e44041114e48e5ade2ca09527db6b01f56cb5271e2c4ef249d7',
  'ja.js': '94bd05ab796608729a0b08616f5ab413d1fe6d8490daf058edad25f67a29387b',
  'ko.js': '0880f50a98a737b22673feb7cbe9fe3233371d0170559a84b413c54529cc90d6',
  'nl.js': '07e4a564472c16ed33064abf8ba45e3750cc81c2312ddac38d20a5bb11ba6f41',
  'pl.js': 'a8ed48e0066b487700fa957ea1329de991cfb1b7d7c2dc9487327ea2c2dbf918',
  'pt.js': '2543c2843e1f32a49d054a734f97da5aafc4b286cbf8f7a6c96c7a5f607ff88e',
  'ru.js': '07439298888e92b8fefa07ab1c5277d74e494a577e6673dffbe0c3c11cb32c22',
  'sv.js': 'd02335571452f63dd061fbef51c9372306d0f554bd41e6d13435119ff5e235c6',
  'th.js': 'a5180ca255919e4e042d53db533af2e8b7bad8053dd7b81785bf997fe5865a41',
  'tr.js': '6f76bdc0bbe183bb082221756135423032f9919a188029527e93034ee2a507af',
  'uk.js': '70bf29024c7889d45e939628585fe26848b3ba409d1877b733e76d24d0cf1f81',
  'vi.js': '38d5f52f1cde6dd77993b44182b96836254acef5c69550141f8982dd38865199',
  'zh.js': 'a67071ad7d6ff556a1db3716d33d694625b91c1d138bd0df72d5342894d1508c',
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
