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

// 4. Fingerprints are pinned to the XAI-LINE-1 pack text (base a0e0f6d9).
var PACKS = {
  'ar.js': 'fb21cac31042a87c93d36f23974b5423488a70a24bdcb62106f3a82276cb7e22',
  'de.js': 'ca0858d5bacc599c66e96e0213b8aa69fa65da5f70b66d209eb36b0b0740bad9',
  'el.js': '229783041e0fb438439d25d3e1721adf1f7675aacd46e8a28896054ce47a6975',
  'en.js': '46364cdaddc31302137d37a1127b0c4c69225b00faa7d8d05517de83d93effe7',
  'es.js': 'b1e2e1a4ae02528ffddb82a0ce4cb3d2b9412f0651b139a72c9a641a919770b1',
  'fa.js': '4ad786084be5c0e6bc5b7768fa21bad43772bb6c5e5e14df575ea98163ddb679',
  'fr.js': '5fdab20f6687906883c86b2a443f4cc85c554c91bc1cf368008917b222f2f405',
  'he.js': 'cc586b67871402f3b0932f940ca8904801346ca6564d132d1194ae2c81846596',
  'hi.js': '1ed3cd8da882bbe975ef67d589a02ccbabde3c3f80d2866cb751968d35e6612d',
  'id.js': '213fb827bce4214525a4c20e882faff05c1c001eb1ed3240192177b4a234dd79',
  'it.js': 'e0282937d6b55af08edfa8996f6a21f1d8b4dd2963d95a994d9e206bc2d3f6fc',
  'ja.js': 'fb095d374fc5d95be0058f6d548a561150c30887f1a09c27bd7e21320ba95415',
  'ko.js': 'bf3f6e30d81d7cf0cf07a17b16e7487dfa3e679d9753c87414754f936c4a69a0',
  'nl.js': '368379c9eef59c45aa6414af054461a68d15b5c43620422bc8c36fa33f43e69f',
  'pl.js': '90f3fba9d456883ea112ac3390a3d5a75295c0a523a1ba7e086e030a1b15ec58',
  'pt.js': 'a4f165414e26333cefff3b27d2e0ea2fd8c71de2278a6978220d457ce7a28f92',
  'ru.js': 'c907d8b938e8a4cbe02267111588ca180fdf22f4333ff3f8b3be0f637c19d11d',
  'sv.js': '40659e25fe52e15b4d07e13a1c69ccfdee2380a2e08a09a0072898c5683cf4e7',
  'th.js': '11e1e45a96c7ef1d358bcde18b393248ef2f352e75aee24b4ed7f448cf1468a2',
  'tr.js': '1cc81d84180db1136e29a003baebdeec034a8fb7c66f40412fa726a51ecd1e41',
  'uk.js': '15ddb147937f3ea669d16197f80325827a79e7b15073aec173691f59c0794a5f',
  'vi.js': '78b92b6181b785c92c99754f0c830a72213d8c57c8ccc23a08d47f41f5368cb0',
  'zh.js': '92fbc76d22bdef1c515db77f03889ea9b31a03ab6f0d101ed8b7a03790e94866',
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

console.log('footer-i18n-1: ' + EXPECT.length + ' footer hooks, keys in all ' + files.length + ' packs, packs unchanged, pilot excluded');
