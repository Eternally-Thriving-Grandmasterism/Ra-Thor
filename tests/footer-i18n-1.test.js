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
  ['homeFooterPrivacy', "'<p class=\"rt-legal\"' + fk('homeFooterPrivacy') + '>The site itself collects no personal data. Computations stay in your browser. No cookies, tracking, or analytics we control. Optional third-party services follow their own policies; licence purchases are covered on the Privacy page.</p>'"],
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
  ['homeMonorepo', "rel=\"noopener\"' + fk('homeMonorepo') + '>Source on GitHub</a>'"]
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
// WP-FAQ-LINK-1: faqA22 links /docs/WHITEPAPER_v4.2.html (label v4.2) in every pack.
// LINKS-FIX-1: English homeMonorepo is "Source on GitHub". Other packs keep their existing translations.
// HOME-ORDER-1: every pack gains standaloneTitle (the home heading that replaced "Start here").
// WHITEPAPER-V43-1: every pack's current-paper mentions move v4.2 → v4.3 (faqA22 link, homePaperLink, weekLine*, homeSurfacePaper);
// English faqA22 label drops "(living cover)"; English homeSurfacePaperNote names v4.3 as current (other packs untouched). No new keys.
// LICENCE-SELF-SERVE-1: privacy and licence keys (homeFooterPrivacy, footerPrivacyText, footerComplianceText, faqA19, privacy*, licence*)
// set to the same new English in every pack (translations pending); browsing still collects no personal data.
var PACKS = {
  'ar.js': '933705ad051270419189c0c08d68992f0cc4ae34aabd54a3c69ff5a44a0ed25f',
  'de.js': '6742e6f92d85c95a63bec493fe066e7d0ffd76b650e11f95db506029f8e50f90',
  'el.js': '57448c4c41faacca41f83f135eb6345eb61851bbb637947d56cf19415202521e',
  'en.js': '4c6b9918bafbe51153ca58d00236c217d079da7deefa4b14006ce8a53f4c93f3',
  'es.js': 'dd8df1f14fb034be59d9e55414b8c6be1da03c675644b08fd6e64a875f96b445',
  'fa.js': '9924476551a162227d52e40aef67baa5c45138e5c63538f3d6d1ddfcc9a6f7b2',
  'fr.js': '3c20bbd552be88231eac11b78e6e1e668e0e4f4f58e828f6454c3a5eba14cf6b',
  'he.js': '17dd708701672b6aa2099c167c11f01a27a904b9df0f0d0ad8bfe7dc11b96504',
  'hi.js': '5cbf293ddf4c59428e0082d3a52d10da87e65a01cc1b5c098dffb4e5a4473eea',
  'id.js': '9787c03ad1aba9e6969e40740cf23b0511a9b03b2338df7159436d5e7601d493',
  'it.js': '89fe74d91211f1d3d2a1414aa08734857f2802289e1dd5fa69d177c1eb8ab5df',
  'ja.js': '4d13a039a5bea64e67d18a1e7fd3cf4101ddda3ac88f833dbbc7edbda80691f7',
  'ko.js': 'd08a47e9d99b2ffe0409ea1f41a7ff1371fb034eeedda0840ac5c48e5bef0936',
  'nl.js': '39fc6d03280ef75543fed973bb00d03c0919874a89671d414ad16735befc06e6',
  'pl.js': 'a0c35254909b11b0c0670fb363e3dc427369bf2ec8f3bce29b659d930a4f8dd3',
  'pt.js': '4d61ea62b3af80ac4ffc13546cdb82c7dd12f5873db981d97f78d5f07920e6ae',
  'ru.js': '092b752d0480ecef1bab4b81d68356809b65ebdb31364d09a9fbef0714891076',
  'sv.js': 'a6a4d352e555617384d2762dd4992c93d82fa93260ab09e7cf55b08dbe90c513',
  'th.js': '828c53c672c5d44e087ac9fa7f055b309e9592718c61b771209752c7cd2384bb',
  'tr.js': '9b398197fcb62a15e0c69b033b20853af4aaac501eb7a7875c551d0ef9f59a95',
  'uk.js': 'd43f459e9fcadf72482d66414a4ff89db05b4645315c5f9eb4e09c7bd1edea38',
  'vi.js': '90405b089f5c4d417c2daffb2af9c694055086f556eea752be7d1f5a8c1859a8',
  'zh.js': 'd9233052b0b633381d2d4fcc1ee533d99080c0f616aa74c9bb08a61bdc9b78bf',
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
