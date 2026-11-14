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
  ['footerTrademarksText', "'<p class=\"rt-legal\"' + fk('footerTrademarksText') + '>AlphaProMega is a trademark of AlphaProMega Media Inc. Powrush, Autonomicity Games and the Autonomicity Games logo are trademarks of Autonomicity Games Inc. The Ra-Thor trademark is pending.<br>Grok is a trademark of xAI. X is a trademark of X Corp.</p>'"],
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
// MODEL-NOTICES-1: licenceFaqQuote carries the English §1B parenthetical in every pack.
var PACKS = {
  'ar.js': '89875bf86c1cd42f0780367faa860914d4dd9106b642f2e20e5be52c5076fb31',
  'de.js': '411f2e562055529b137861faad236fb5e4a69c2faba50a8d4161b23e08793db2',
  'el.js': 'f740cb9d09153038306724cef37e2f337f622ecdd6ff6c3dfaece819e12b692d',
  'en.js': '2cfe1e6d28c5d8e332c8a686ab935f572a395e17e5c681a97a86375082ec1998',
  'es.js': '753c0d847d294405fe3fb187d8d03dd56b4ef989528d06729710f70f5b5ece93',
  'fa.js': 'a4331e3e4bc82e69d631cb930be3e3ed39be277f448dbc242811e661ea9af6e6',
  'fr.js': 'd68480ec233e17119c046f0bf3b586e238a45b9be3e18df008c6c5f5579d589f',
  'he.js': 'c1c0eb607aa2aa4f09eba650982aa545b60dae94edf08a59e893fa96d318f7d0',
  'hi.js': '209748c69611500edb340e0cc0f9982f511497550fafc98a4eb73dd3182f5073',
  'id.js': '59f378ad9666e902ab72928cb022291796555bb4b4d9c45e2446546010f642a3',
  'it.js': 'bcae631751553a902d216db7beebf87ef60da69068ef9e5c80e6761e3cc100fb',
  'ja.js': '27f876fa93652d317078ced7a7ca996bd76a8ad34969329ebbf3e31d0447b4fa',
  'ko.js': 'bc8c8ecdb8bbf6b2ef0d140b9cc6d5b5235c81c542d4e56b272b2a9aa13f0070',
  'nl.js': '91a42d91406f548496a835af0fe00ab7459a326d3572680d692ecd503b88c3e0',
  'pl.js': '310fc5c842cdc40b2e33d0214d2a5268cee9f74def944624f5589e4d3feeec04',
  'pt.js': 'c5c7190cc61a3ec1e60ec1c3578ae07b6ffc1abb8bb3909616f0b26282e9fe7f',
  'ru.js': '9c17ef3ca42371255c2e85b01be975a513723a2dbf1dedbaf6967211fdd71976',
  'sv.js': '8dc7100bf7c255bccf24113840cb2f67b8880944c0be5f0ff12f5411ca1d23ed',
  'th.js': '6c3d6750d8fee7c0aa2db442bd38121fac45e170195e7a5bd9bfca0e43a22fae',
  'tr.js': 'e15843a84bd04c9b5876c00db5e8379e6cf9f5bb95a6a497adf19059666e7c3a',
  'uk.js': 'b41862171cd6338fbbd0041531bd023ba191460861e1356fc43a7a77556c3c16',
  'vi.js': '42dc610c5e83f47eb59cad1a8cdcc96ce94b79224b36e9ccf4cdbe47202450f0',
  'zh.js': '5b967ed6e447b5584998e361a265b5619db9e302fc24632694e92a6a1446904e',
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
