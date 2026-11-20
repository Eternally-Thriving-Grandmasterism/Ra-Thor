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
  ['footerTrademarksText', "'<p class=\"rt-legal\"' + fk('footerTrademarksText') + '>AlphaProMega is a trademark of AlphaProMega Media Inc. Powrush, Autonomicity Games and the Autonomicity Games logo are trademarks of Autonomicity Games Inc. Ra-Thor™ is a trademark of Autonomicity Games Inc. (application pending). <br>Grok is a trademark of xAI. X is a trademark of X Corp.</p>'"],
  ['footerPrivacyTitle', "'<h4' + fk('footerPrivacyTitle') + '>Privacy</h4>'"],
  ['homeFooterPrivacy', "'<p class=\"rt-legal\"' + fk('homeFooterPrivacy') + '>Browsing this website collects no personal data; payments are processed by Stripe. Computations stay in your browser. No cookies, tracking, or analytics we control.</p>'"],
  ['footerWorkspaceTitle', "'<h4' + fk('footerWorkspaceTitle') + '>Workspace</h4>'"],
  ['homeFooterWorkspace', "'<p class=\"rt-legal\"' + fk('homeFooterWorkspace') + '>v14.15.6 · AG-SML v1.2 · TOLC 8 · Capable · Bounded · Corrigible</p>'"],
  ['footerFamilyTitle', "'<h4' + fk('footerFamilyTitle') + '>Directory</h4>'"],
  ['navHome', "'<a href=\"/\"' + fk('navHome') + '>Home</a>'"],
  ['employTitle', "'<a href=\"/employ.html\"' + fk('employTitle') + '>How to employ</a>'"],
  ['navPilot', "'<a href=\"/pilot.html\"' + fk('navPilot') + '>Pilot</a>'"],
  ['homeLaunchMap', "'<a href=\"/Launch-Ra-Thor.html\"' + fk('homeLaunchMap') + '>Launch map</a>'"],
  ['homeMoments', "'<a href=\"/micro-moment.html\"' + fk('homeMoments') + '>Micro-moments</a>'"],
  ['surfaceShard', "'<a href=\"/sovereign-shard.html\"' + fk('surfaceShard') + '>Sovereign Shard</a>'"],
  ['navContact', "'<a href=\"/contact.html\"' + fk('navContact') + '>Contact</a>'"],
  ['navPrivacy', "'<a href=\"/privacy.html\"' + fk('navPrivacy') + '>Privacy</a>'"],
  ['navExamples', "'<a href=\"/examples.html\"' + fk('navExamples') + '>Examples</a>'"],
  ['navLicence', "'<a href=\"/licence.html\"' + fk('navLicence') + '>Ra-Thor Commercial Licence</a>'"],
  ['navLineage', "'<a href=\"/lineage.html\"' + fk('navLineage') + '>Lineage</a>'"],
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
// Hashes updated by PRIVACY-DATE-1: privacyUpdated → 2026-10-22, employStamp → 22 Oct 2026, swatchStamp date stripped.
// MODEL-NOTICES-1: licenceFaqQuote and faqA16 carry the same English §1B parenthetical in every pack.
var PACKS = {
  'ar.js': '59fb2ede37ebadf1d006a2e0ec58d5a077e7873d742f859f3910fd77042d768f',
  'de.js': 'e6ae9700d1e377d9f0939a5b53271a200818d79b795b27e93b82ed872a2a6ba4',
  'el.js': '171081afd695ece40ad2541ce58b18f87215229645078043817b4bc20377ac17',
  'en.js': '6ccefc72e8e4d2cb2957ad962d657fe40863ae38905dd57b65d23c623f8f7714',
  'es.js': 'fe7eb0f6766a1017aab12a1730d15d961f247753ee1f68d49acdf020459ab6dc',
  'fa.js': '6a79693746f3ca4d673c4e1335fc383e57913b636fb66982c6be06a43bef8cfe',
  'fr.js': '9fb31fde894781af5234060d189e60a3e708af3cb41c9fd1710e0f6d8a4a09b7',
  'he.js': '7b3421322c4e294b0baedca188e10a06451756b9eebdf7ca2d42205fa7603342',
  'hi.js': 'da13dcb2554cbddee23a8877c04d82dc21cd6e6a863a7ebedf5675597630ee12',
  'id.js': 'b595a4782736ccfe3934556dfc28203fd7eb4060c2d089f204c5b42b920ca694',
  'it.js': 'e3591be9feb0426ef3a898fe820cf4b7c0da415f2fe0525e29e39e5bb7a67df9',
  'ja.js': '0a2ff6e26a9bfcc7b8927f8f7c36758261e4696f1f8c56f682668183a637b26f',
  'ko.js': 'dfec6d22455f72d9c9bcb744c07e95d63ea9cb5d78febd02ccfa0c3e050296d4',
  'nl.js': 'edc35623e4bf37bff801ef53e43df98e800d893e787277755d69f009afbefbe9',
  'pl.js': '19f61fd744e137b78a4c06d2357081163c5052714c191c693d164702cf6a1353',
  'pt.js': 'f54372f11631c9f08585def4628518db35d65a87df826b26b6344c62831c71a2',
  'ru.js': '7b456ce7a7293d2e64fbf6349c18b2a4a74dbe4968a5c538c89d55f0bf2f1419',
  'sv.js': '4348868d26e687c1650cd6d9326bba4d6ea04c38096236f3ad45c0bae9c9fa89',
  'th.js': '3d28a5106ba6c33aca50b1b8407e3d5df9d0c1fc0c65f76e4f4a773d47297592',
  'tr.js': '309976b2a01b76adfcd62adea0da5589a895abc7c1700cb589a3d4af22a98338',
  'uk.js': 'df69e2be2430bb4f9b241f3a20c92cc2ba25334819827e2a18ed96a31d381079',
  'vi.js': '8f4715bfef927928c60475868ca167d89e0a3288b3dbfd13094ef9e8794fdd16',
  'zh.js': 'bc4740d814615667b67271c9c4b6b71d6baf09674f2d85c80df5a4d0f96e8d5f',
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
