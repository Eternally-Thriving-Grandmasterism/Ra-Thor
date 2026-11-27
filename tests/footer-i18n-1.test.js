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
  ['homeFooterPrivacy', "'<p class=\"rt-legal\"' + fk('homeFooterPrivacy') + '>Browsing this website collects no personal data. Computations stay in your browser. No cookies, tracking, or analytics we control. Licence purchases are covered on the Privacy page.</p>'"],
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
  'ar.js': 'a280c126d09fb722d840a09dc87532c3969905cff37eb6b00e61fcafe409e65d',
  'de.js': '7f11688db3ffdea549b331ac80fb4826be92ea08405bae0341eea15ce62635de',
  'el.js': 'f106f5c4088a6be7dbe7a6439afd1ec4ee2f37491e9fdba7448fb6d616ea276e',
  'en.js': 'fd7f418c44e667d1bcbe64ea57197775781026104fafb32cd965dadb7dbed3f2',
  'es.js': 'eb29b8ce053430cf8efcd5e085d7d06dc8f8098923a9d11e42a506130c433289',
  'fa.js': '1cfc546ea82fb7521035f56f9bc02a15ac1cbc5efd2fedd74e9199f4900feba0',
  'fr.js': '22fe6f13ea1cc5323d89aba99a250ddf08de79acae60cd8a6b3e9d86bd507b33',
  'he.js': '295a3f93dc68c7668396fd336c10719ee997e02d0ac09b152d873a26e3c502d0',
  'hi.js': '4f9c3da4f4aeffe935ad5cd852b0294c062778e59394bac1b7244456f90ae91c',
  'id.js': '61d0ab71ec9db56db1265bab42477ebd480f145613937968f6b1e663de5aea7a',
  'it.js': '77050bc9e55ce48136144c034da8026cbf6606f92b79703843317820ed72cd06',
  'ja.js': '298e3d573db5fcd7dd961101bb5015a5acf7b4ea0907d20c7e830d75aa418e51',
  'ko.js': 'a8512b9c065153f3f7724e85de669f666d2b784ba4111fc56332a87310695bac',
  'nl.js': 'f2787bf7b429b85f980170923b8555a79ef5f786c02b2cb29e5d492ca6da27e2',
  'pl.js': 'd240bb09ae6028b0ea0f8a147c3a800640685c95bd0b1605b167fa374bb3ef71',
  'pt.js': '76f350235f74cae499dadccd18ff9f9f7a5bff4747611d89fddfcf97af4d1aa5',
  'ru.js': '43b7fd9a377ba51f85420c9cf9f9641fe65f2dcc113da46efeb20e1c507598f5',
  'sv.js': 'b800fb4763770bd7106bab2c04e8721fef4889521f31572adee98102f62107ef',
  'th.js': '3f891192b325464389f0ed147d33dbcf377211f02ebc2c77fdf83badbaf71190',
  'tr.js': '460471255c3e74be9e82189344fbedc7bb651712749a22c858a6a3cacc031d3d',
  'uk.js': '35c905eab4b6c276488d6439bfae8dc07d45affc1117574a70c1fd142384dbf8',
  'vi.js': 'af4141b3d39f084505265b191934a06c7b33d80504a48203c7a56c10187791af',
  'zh.js': 'aa3aae21467e2be15c21e2d8d3f7e308699fa3b5036dd79626073d5742192cf2',
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
