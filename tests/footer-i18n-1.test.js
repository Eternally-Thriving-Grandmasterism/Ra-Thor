/* FOOTER-I18N-1: the shared fallback footer (js/family-nav-2026-08-22.js siteFooter())
 * follows the chosen language using EXISTING pack keys only.
 * Checks: the inserted footer elements carry the pack keys, those keys exist in all
 * 23 packs, no pack file changed (sha256 pinned at main d6b17afd), English text is
 * unchanged, and /pilot.html gets no hooks. */
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
  ['footerTrademarksText', "'<p class=\"rt-legal\"' + fk('footerTrademarksText') + '>Ra-Thor™ is a trademark of Autonomicity Games Inc.<br>Grok is a trademark of xAI. X is a trademark of X Corp.<br>Ra-Thor is independent — not affiliated with, sponsored by, or endorsed by xAI.</p>'"],
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
 '<div>© 2026 Sherif Samy Botros — sole steward of Autonomicity Games Inc. & AlphaProMega Air Foundation. TOLC 8 · independent of xAI.</div>'
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

// 4. No pack file changed (sha256 of main d6b17afd).
var PACKS = {
  'ar.js': '69e3fcd45e76b56813f8915a9d89fc4a901cd1e1592278821da78de7183fa382',
  'de.js': 'c67ab424781f436f678cfa3d2c64e7046d3d84ca66d4534b5cbbaad3ad9efaf4',
  'el.js': '64cffd8488935fe84b1c0ed7d2b3c426d7a228df337f8f845585b4a3a9b090b0',
  'en.js': '7953e2a7626220a9a67a3ed69ff0d1fb6934ce1dd4dbef43f679e46df7e336b1',
  'es.js': 'c7050244e05f448621642bdb6e2a71baf0333e023b0f26ed81f2dcb2e979a80f',
  'fa.js': 'af71d19fd4fc9ec49b26490f14f0ec08bbf79a3782c52a600992637345ad3345',
  'fr.js': '8620081a1c996f70f85927ab8f7b8c1a0aaa90ef5027e1c3c90116d1acc79fef',
  'he.js': 'd715f02c04feffc060bda8f3d411522291746cfb23aa66fcbdb39de9240abd05',
  'hi.js': '23c6667f6b03e890f886d8c63921a37258cae132dba445a87d03c743badf6f45',
  'id.js': 'fd4abf134d8ea29550c75d363a7847de44c6a3d67e19eab52cc4136248f158ca',
  'it.js': '89fb447072e1f648f272c5297ce95ccfd3c4762f4bb9f8c5a7359f022dac728b',
  'ja.js': '06560a1d6effed272b996f60e53d6c8a3f1d1d92a6d0d209c87e74d332b6ff2b',
  'ko.js': 'dfc0789d287bc1c97456929b2242beccf2335ccdf3900e5746d43539649692e4',
  'nl.js': '0f9417d8f94f3370ef1ed68558c10d09a5752b3fe4189dfa14098ff2cb46ec09',
  'pl.js': '526e3a963a540cea0b0e72e44ab9be09f579b264f0024d8137ec381242705c7b',
  'pt.js': '11ca02928a394484c61d95aed96077d51559824c0f3f4fe1ba89ad58cb3fcfb3',
  'ru.js': 'db9cd87d1a4e41d71c137b056784298763ca1b11c0ac6d1d214a52b473d3d9f6',
  'sv.js': '8282d59f44291f7f2e3a1bc65f851eea1447e1d1b15e4ae85c157904392547b4',
  'th.js': '738f103498b53962b375fa9732ec499f02481cd8b6920898d15d49c829bc571a',
  'tr.js': '6d47e75d925d8aa95fba5c465f4dae32825bab450b16eb0cf0e3c9dacb23b583',
  'uk.js': '096e91b26e3f2ed8a3f63982c05913974c93922e92cac13505bfd30460c0bc3f',
  'vi.js': '9eb992a450f90057b04b68a04b7d2b11a47d725b4372a11237f88f3eccc60da4',
  'zh.js': 'b9fce163437fed97335ad6073c0811c0d943f867e6d4206e5c8e0bb24ae29909',
};
assert(Object.keys(PACKS).length === 23, 'pack manifest must list 23 packs');
files.forEach(function (f) {
  var h = crypto.createHash('sha256').update(fs.readFileSync(path.join(root, 'i18n', f))).digest('hex');
  assert(PACKS[f] === h, 'pack changed: i18n/' + f);
});

console.log('footer-i18n-1: ' + EXPECT.length + ' footer hooks, keys in all ' + files.length + ' packs, packs unchanged, pilot excluded');
