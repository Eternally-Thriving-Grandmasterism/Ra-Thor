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
  ['homeFooterPrivacy', "'<p class=\"rt-legal\"' + fk('homeFooterPrivacy') + '>This website collects no personal data. Computations stay in your browser. No cookies, tracking, or analytics we control.</p>'"],
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
var PACKS = {
  'ar.js': '49e97fb56c97fe5c48e087f301973a34e58eacbc6757827b0e76c8d7079de698',
  'de.js': 'd4b7e09d0788d96beb79b2e56a97b780c7d0d4cd6d910b89b3d2e57a3975de9a',
  'el.js': 'f9c85d5a80593d9649a7962297dff81305ac74ad5121055609f039e1b8097bbb',
  'en.js': '6b76acf70f2f79fdc30323895c711247d2960c24a384d111ee59b90acf003f6e',
  'es.js': 'c2d97c17ae4b6e6d3c233d00952a0a0d999b06ece5b956edb87612bd2864ad60',
  'fa.js': 'ec3fafd0220d9614fbe6f74dd1e7ce155f07300c5ecb4eaa08fa28e494a19987',
  'fr.js': 'db2b4be1b710b87cef9648cea33ade4461cbe5cb9c1b467f3f123d972ccbce6f',
  'he.js': '48eb19639a9c271c1213abbd417b65ac1231b1e729097c526d9d503e229fd580',
  'hi.js': '4eaebf9946b16ec02c482f9487095170ff137947b5791f0a0114077c05ec1e58',
  'id.js': 'e2c0e3e8f417655d58e843b3c3762206cc297b35e8f25f484c63516d22ed9c38',
  'it.js': '55fa8064096c6a71ec01d6e790b7a73ebcd52dffb7306f45ecde795abd13a8fd',
  'ja.js': 'a4c709400c0e99e143db585b5ae57484a90788400c4c951faff77456234acf0d',
  'ko.js': '17fe38758f81a8cc1cc24c036ec8788344744e832a337ceac9837e009609c12b',
  'nl.js': '5510019af6fcfd3fb33af8d79873c4b0ea6d9153c6bbce95aeba785a55ef078c',
  'pl.js': '1ef38afeb19a2e8b6571a71e47e1eff7f36b150400fd44e1a76699051d176972',
  'pt.js': 'c5b4c4241e69095d130631120bd245b56e18bcbeda03901f5c5b995790852c96',
  'ru.js': '42fa210deefba96a6bb2dd2fc1221d6d6f7cb558851d9d657bcc38ad27feaee9',
  'sv.js': '165c3750017b9e52ecec42396538a960c67b07e6ce2408ee7a85bb016bef6acc',
  'th.js': 'ff9f69110783e81683d74be1c1a0c7c460030f1206e26e4f17f3d10a3bc7f26f',
  'tr.js': 'e76abff331047e49a9871d6250bbbaf793cf1ffe3f149afa585f26757c149f45',
  'uk.js': '055e35cdd73f539d1f14acf57bcd9b50a2c66be2a6a63d4d76295f047ae5e528',
  'vi.js': 'c3c63b544addb930811f193e2ecb97016d5d0c8db9c78bc1c2f4fb1af215b09c',
  'zh.js': 'd2300011ca614b972c7bc0a07929094d553d802f7af6c180052f557eee0935b8',
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
