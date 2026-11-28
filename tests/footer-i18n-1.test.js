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
  'ar.js': '1bc1abf20dbd8da73dee0a3f4d3eb1140b8610818b7f21325548bf4f47d4e848',
  'de.js': '0b34a0da0af7a1a21cb0ee13305c0196928047357d0aa73ff192b3916c390d79',
  'el.js': '05215f73cb6f95bb032ccbf6b98426e6c13eb82d3118a016ead5c9b8952a815f',
  'en.js': 'eb7928305effc9c2a320fe40d33449815eb5575f11d7bd67de799f60e9f5ae29',
  'es.js': '2b96e347b06b22ab894cfbf2cf42d0afb2199f3391acee082b7165ff984300aa',
  'fa.js': '8c9ac9c8bb7afd20cba76dccd4fce4016c473c613e4964021a21f0b8fe7fd1f6',
  'fr.js': '08abe7c4959f20d7c866d7bf6350ba86d50fda69497fce3144858793098b1193',
  'he.js': 'e07fc7fd1f93af5e45513c00fa7bcb968009adae0083bf7544a19bb99de52de0',
  'hi.js': 'd15813ee2027aaabc178425180e497462638a92f22b30ba0bc5f4a8a6631e7f6',
  'id.js': '43f6392d8f162e059a4043916a499312f7f73aa990ce577d745d8ac03956ceec',
  'it.js': 'baeef737216292df9cce87f2a787ed1fcd27f8667a27645da7ec1d2c33facdfb',
  'ja.js': '545271eb9c497bb2402a16c4c6bfd89e0914ecd55272c14b05332eeffd6af952',
  'ko.js': 'f41233b3862cf260852032edf6612615263c8ec3c25c328ba660c4d592512a31',
  'nl.js': '128e72f56ba0adca74e946beaa2bfafe416696bedf1df376069e09ac2b859451',
  'pl.js': '9c2ced994e75783a54da5aa188543be899f433efb1b6f8d3a03b7a6c8d62a678',
  'pt.js': 'effa82c22a364b4a44677c3ec5b34f66a849806c86c0e741acc3c8f4017b67bb',
  'ru.js': '686343c1e031257a7279a0b521ae5175552381a5b9bf181cf519e392583016b7',
  'sv.js': '4d792fad9babbbf5db0e32b187acaec964bbaae9b8144e7a089a1cf3f9cfe368',
  'th.js': 'c2d5059539c0df15015694a503ced8635aa72331f3fc17a68c7e25f2c78f9c49',
  'tr.js': 'fa2b2c0e348270f3acb9c047255f8e8a52a9e8626c57250c1a2661a510f53c91',
  'uk.js': '2e84ef25fccabfce3f1077a3c9657f2c3a8e04074de581fa7fd06ec73b78a3a9',
  'vi.js': 'b0a99e8e8e4b5ddf663940a6f78e8bf8683d4d8dc8b569a318841b42f0129d74',
  'zh.js': '4c5e4ffe6f84e3ac2db305f1f86750ed27d32f627b0a6f2a8808f1984e83b438',
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
