/* EMPLOY-1: stranger employ spine. One whitepaper; living pages follow it. */
var fs = require('fs');
var path = require('path');
var root = path.join(__dirname, '..');
var employMd = fs.readFileSync(path.join(root, 'docs/EMPLOY.md'), 'utf8');
var employHtml = fs.readFileSync(path.join(root, 'employ.html'), 'utf8');

function assert(cond, msg) {
  if (!cond) throw new Error(msg);
}

assert(employMd.indexOf('inspect') !== -1 && employMd.indexOf('METR') !== -1, 'docs/EMPLOY.md must contain inspect / METR');
assert(employMd.indexOf('info@Rathor.ai') !== -1, 'docs/EMPLOY.md must contain info@Rathor.ai');
assert(employMd.indexOf('14.15.6') !== -1, 'docs/EMPLOY.md must contain 14.15.6');
assert(employMd.indexOf('Layer 0') !== -1, 'docs/EMPLOY.md must contain Layer 0');
assert(employMd.indexOf('cargo test --workspace') !== -1, 'docs/EMPLOY.md must name cargo test --workspace');
assert(employMd.indexOf('Powrush-MMO is a separate repo') !== -1, 'docs/EMPLOY.md must keep dual-repo');
assert(employMd.indexOf('independent of xAI') !== -1, 'docs/EMPLOY.md must say independent of xAI');

assert(employHtml.indexOf('article') !== -1 && /<article\b[^>]*\brt-prose\b/.test(employHtml),
  'employ.html must keep article.rt-prose');
assert(employHtml.indexOf('mailto:info@Rathor.ai') !== -1, 'employ.html must keep mailto:info@Rathor.ai');

assert(employMd.indexOf('## The loop') !== -1, 'docs/EMPLOY.md must restore The loop');
assert(employMd.indexOf('Intend') !== -1, 'docs/EMPLOY.md loop must name Intend');
assert(employHtml.indexOf('Pass the gates') !== -1, 'employ.html must restore Pass the gates');
assert(employHtml.indexOf('14.18') === -1, 'employ.html must not sell 14.18');

var optionalMd = fs.readFileSync(path.join(root, 'docs/OPTIONAL_MODEL.md'), 'utf8');
var skillMd = fs.readFileSync(path.join(root, 'skills/ra-thor-employ/SKILL.md'), 'utf8');
assert(optionalMd.indexOf('inspect') !== -1 && optionalMd.indexOf('METR') !== -1, 'OPTIONAL_MODEL.md must contain inspect / METR');
assert(optionalMd.replace(/\*/g, '').indexOf('not an xAI product') !== -1, 'OPTIONAL_MODEL.md must keep Grok optional');
assert(optionalMd.indexOf('/v1/chat/completions') !== -1, 'OPTIONAL_MODEL.md must name the OpenAI-compatible door');
assert(optionalMd.indexOf('AgentOS-certified') !== -1, 'OPTIONAL_MODEL.md must refuse AgentOS-certified claim');
assert(skillMd.indexOf('name: ra-thor-employ') !== -1, 'SKILL.md must use agentskills name');
assert(employMd.indexOf('OPTIONAL_MODEL.md') !== -1, 'docs/EMPLOY.md must point at OPTIONAL_MODEL.md');
assert(employHtml.indexOf('OPTIONAL_MODEL.md') !== -1, 'employ.html must point at OPTIONAL_MODEL.md');
assert(employHtml.indexOf('certified AgentOS') !== -1, 'employ.html must refuse certified AgentOS');

var adoptMd = fs.readFileSync(path.join(root, 'docs/ADOPT.md'), 'utf8');
var wrapPy = fs.readFileSync(path.join(root, 'wrappers/local-shim/rathor_wrap.py'), 'utf8');
var sysPrompt = fs.readFileSync(path.join(root, 'wrappers/system-prompt.txt'), 'utf8');
assert(adoptMd.indexOf('There is **no** public rathor.ai proxy') !== -1, 'ADOPT.md must refuse public key proxy');
assert(adoptMd.indexOf('inspect') !== -1, 'ADOPT.md must keep inspect claim');
assert(wrapPy.indexOf('/v1/chat/completions') !== -1, 'shim must name chat completions');
assert(wrapPy.indexOf('RATHOR_UPSTREAM') !== -1, 'shim must use operator upstream');
assert(sysPrompt.indexOf('14.15.6') !== -1, 'system-prompt must name workspace');
assert(sysPrompt.indexOf('info@Rathor.ai') !== -1, 'system-prompt must name contact');
assert(employMd.indexOf('ADOPT.md') !== -1, 'EMPLOY.md must point at ADOPT.md');
assert(employHtml.indexOf('Local HTTP wrap') !== -1, 'employ.html wrap card must name Local HTTP wrap');
assert(employHtml.indexOf('docs/ADOPT.md') !== -1, 'employ.html wrap card must point at docs/ADOPT.md');
assert(employHtml.indexOf('rathor_wrap.py') !== -1, 'employ.html wrap card must link rathor_wrap.py');
assert(employHtml.indexOf('There is no public rathor.ai key proxy') !== -1, 'employ.html wrap card must refuse public key proxy');

assert(employHtml.indexOf('Do not change the walk') === -1, 'employ.html must not print operator HOLD walk');
assert(employHtml.indexOf('Do not invent pages') === -1, 'employ.html must not print operator HOLD doors');
assert(employHtml.indexOf('Chrome-only i18n') === -1, 'employ.html must not print Chrome-only i18n');
assert(employHtml.indexOf('ceo@acitygames.com') === -1, 'employ.html must not print ceo@acitygames.com');
assert(employHtml.indexOf('14.15.6') !== -1, 'employ.html must name workspace 14.15.6');
assert(employHtml.indexOf('inspect') !== -1, 'employ.html must keep inspect');
assert(employHtml.indexOf('info@Rathor.ai') !== -1, 'employ.html must name info@Rathor.ai');
assert(employHtml.indexOf('Outputs are drafts') !== -1, 'employ.html must keep Outputs are drafts');
assert(employHtml.indexOf('Intend') !== -1, 'employ.html loop must name Intend');
assert(employHtml.indexOf('Act') !== -1, 'employ.html loop must name Act');
assert(employMd.indexOf('Public voice lives on employ.html') !== -1, 'docs/EMPLOY.md must keep public voice on employ.html');
assert(employMd.indexOf('Do not change the walk') !== -1, 'docs/EMPLOY.md must keep operator HOLD walk');
assert(employMd.indexOf('Do not invent pages') !== -1, 'docs/EMPLOY.md must keep operator HOLD doors');
assert(employMd.indexOf('Chrome-only i18n') !== -1, 'docs/EMPLOY.md must keep Chrome-only i18n HOLD');
assert(employMd.indexOf('ceo@acitygames.com') !== -1, 'docs/EMPLOY.md must keep deprecated-address HOLD');

assert(wrapPy.indexOf('stream=false only') === -1, 'shim must no longer refuse stream');
assert(wrapPy.indexOf('text/event-stream') !== -1, 'shim must byte-forward SSE as text/event-stream');
assert(wrapPy.indexOf('/v1/models') !== -1, 'shim must expose GET /v1/models');
['gemini.md', 'cursor.md'].forEach(function (name) {
  var snippetPath = path.join(root, 'wrappers/custom-instructions', name);
  assert(fs.existsSync(snippetPath), name + ' must exist');
  var snippet = fs.readFileSync(snippetPath, 'utf8');
  assert(snippet.indexOf('14.15.6') !== -1, name + ' must name workspace 14.15.6');
  assert(snippet.indexOf('info@Rathor.ai') !== -1, name + ' must name contact info@Rathor.ai');
});

var chatJs = fs.readFileSync(path.join(root, 'js/chat.js'), 'utf8');
assert(chatJs.indexOf('14.15.6') !== -1, 'js/chat.js SYSTEM_PROMPT must name workspace 14.15.6');
assert(chatJs.indexOf('inspect') !== -1, 'js/chat.js must contain inspect');
assert(chatJs.indexOf('AGSi demonstration') === -1, 'js/chat.js must not sell AGSi demonstration');
assert(chatJs.indexOf('symbolic AGI lattice') === -1, 'js/chat.js live prompt must not sell symbolic AGI lattice');
assert(chatJs.indexOf('AG-SML v1.0') === -1, 'js/chat.js live prompt must not sell AG-SML v1.0');
assert(chatJs.indexOf('Outputs are drafts') !== -1, 'js/chat.js Copy Context / SYSTEM_PROMPT must quote drafts');
assert(chatJs.indexOf('Independent of xAI') !== -1, 'js/chat.js Copy Context / SYSTEM_PROMPT must quote independent of xAI');
assert(chatJs.indexOf('SYSTEM_PROMPT.trim()') !== -1, 'Copy Context must quote SYSTEM_PROMPT (same sentences)');

var briefingMd = fs.readFileSync(path.join(root, 'docs/PUBLIC_EMPLOY_BRIEFING.md'), 'utf8');
var briefingHtml = fs.readFileSync(path.join(root, 'briefing.html'), 'utf8');
assert(briefingMd.indexOf('14.15.6') !== -1, 'briefing markdown must name workspace');
assert(briefingMd.indexOf('info@Rathor.ai') !== -1, 'briefing markdown must name contact');
assert(briefingMd.indexOf('inspect') !== -1 && briefingMd.indexOf('METR') !== -1, 'briefing markdown must keep inspect / METR');
assert(briefingMd.replace(/\*/g, '').indexOf('no public rathor.ai') !== -1, 'briefing markdown must refuse public key proxy');
assert(briefingHtml.indexOf('14.15.6') !== -1, 'briefing.html must name workspace');
assert(briefingHtml.indexOf('info@Rathor.ai') !== -1, 'briefing.html must name contact');
assert(briefingHtml.indexOf('/employ.html') !== -1, 'briefing.html must point back at Employ');
assert(briefingHtml.indexOf('Do not change the walk') === -1, 'briefing.html must not print operator HOLD');
assert(employHtml.indexOf('/briefing.html') !== -1, 'employ.html must link the public briefing');
assert(employMd.indexOf('PUBLIC_EMPLOY_BRIEFING.md') !== -1, 'docs/EMPLOY.md must point at the public briefing file');

var sisterMd = fs.readFileSync(path.join(root, 'docs/SISTER_ADOPTION.md'), 'utf8');
assert(sisterMd.indexOf('## LINK') !== -1, 'SISTER_ADOPTION.md must keep LINK');
assert(sisterMd.indexOf('## EXTRACT later') !== -1, 'SISTER_ADOPTION.md must keep EXTRACT later');
assert(sisterMd.indexOf('## LEAVE') !== -1, 'SISTER_ADOPTION.md must keep LEAVE');
assert(sisterMd.indexOf('members =') === -1, 'SISTER_ADOPTION.md must not add Cargo members');
assert(sisterMd.indexOf('Ra-Thor ships fusion') !== -1, 'SISTER_ADOPTION.md must refuse fusion SKU');
assert(sisterMd.indexOf('AG-SML v1.1') !== -1, 'SISTER_ADOPTION.md must name living grant v1.1');
assert(sisterMd.indexOf('info@Rathor.ai') !== -1, 'SISTER_ADOPTION.md must name info@Rathor.ai');
assert(employHtml.indexOf('SISTER_ADOPTION.md') !== -1, 'employ.html must point at SISTER_ADOPTION.md');
assert(employMd.indexOf('SISTER_ADOPTION.md') !== -1, 'docs/EMPLOY.md must point at SISTER_ADOPTION.md');
assert(employHtml.indexOf('Do not add sister') === -1, 'employ.html must not print operator sister HOLD');

var indexHtml = fs.readFileSync(path.join(root, 'index.html'), 'utf8');
assert(indexHtml.indexOf('Eternal Mercy Thunder') !== -1, 'index.html must restore Eternal Mercy Thunder');
assert(indexHtml.indexOf('fonts.googleapis.com') === -1, 'index.html must not load fonts.googleapis.com');
assert(indexHtml.indexOf('ceo@acitygames.com') === -1, 'index.html must not contain ceo@acitygames.com');
assert(indexHtml.indexOf('id="hero-headline"') !== -1 && indexHtml.indexOf('title-font') !== -1 && indexHtml.indexOf('thunder-glow') !== -1, 'hero must keep title-font + thunder-glow');
assert(indexHtml.indexOf('Delivered software') === -1, 'index.html must not restore Delivered software');
assert(indexHtml.indexOf('WHITEPAPER_v4.2') !== -1, 'index.html must point at living cover v4.2');
assert(indexHtml.indexOf('WHITEPAPER_v4.1') === -1, 'index.html must not present v4.1 as the living cover');
assert(indexHtml.indexOf('Whitepaper v4.1') === -1, 'index.html must not advertise v4.1 as the living cover');

var restB = fs.readFileSync(path.join(root, 'css/rathor-theme-rest-b.css'), 'utf8');
assert(restB.indexOf('https://fonts.googleapis.com') === -1, 'theme CSS must not load fonts.googleapis.com');
assert(restB.indexOf('"Cinzel", "Cinzel Decorative", Palatino, "Palatino Linotype", "Times New Roman", serif') !== -1, 'theme .title-font must use the Cinzel stack');
assert(restB.indexOf('url("/fonts/cinzel/Cinzel-Regular.woff2")') !== -1, 'theme must self-host Cinzel Regular');
assert(restB.indexOf('url("/fonts/cinzel/Cinzel-Bold.woff2")') !== -1, 'theme must self-host Cinzel Bold');
assert(fs.existsSync(path.join(root, 'fonts/cinzel/Cinzel-Regular.woff2')), 'Cinzel Regular woff2 must exist');
assert(fs.existsSync(path.join(root, 'fonts/cinzel/Cinzel-Bold.woff2')), 'Cinzel Bold woff2 must exist');
assert(fs.existsSync(path.join(root, 'fonts/cinzel/OFL.txt')), 'Cinzel OFL.txt must exist');
assert(fs.readFileSync(path.join(root, 'fonts/cinzel/OFL.txt'), 'utf8').indexOf('SIL Open Font License') !== -1, 'OFL.txt must be the SIL OFL');

var employTitle = employHtml.match(/<h1[^>]*data-i18n="employTitle"[^>]*>/);
assert(employTitle, 'employ.html must keep employTitle h1');
assert(employHtml.indexOf('Cinzel') === -1, 'employ.html must not name Cinzel');
var privacyHtmlForFont = fs.readFileSync(path.join(root, 'privacy.html'), 'utf8');
assert(privacyHtmlForFont.indexOf('Cinzel') === -1, 'privacy.html must not name Cinzel');
var familyNav = fs.readFileSync(path.join(root, 'js/family-nav-2026-08-22.js'), 'utf8');
assert(familyNav.indexOf("{ href: '/employ.html', label: 'Employ' }") !== -1, 'family bar must keep Employ');
assert(familyNav.indexOf('Cinzel') === -1, 'family pills must not switch to Cinzel');

var privacyHtml = fs.readFileSync(path.join(root, 'privacy.html'), 'utf8');
assert(employHtml.indexOf('family-nav-2026-08-22.js') !== -1, 'employ.html must load family-nav-2026-08-22.js');
assert(privacyHtml.indexOf('family-nav-2026-08-22.js') !== -1, 'privacy.html must load family-nav-2026-08-22.js');
assert(employHtml.indexOf('Do not change the walk') === -1, 'employ.html must not contain Do not change the walk');

var familyPages = [
  'index.html', 'employ.html', 'privacy.html', 'chat.html', 'contact.html',
  'Launch-Ra-Thor.html', 'micro-moment.html', 'sovereign-shard.html',
  'web-forge.html', 'briefing.html', 'constellation-week.html', 'go-x.html'
];
familyPages.forEach(function (name) {
  var html = fs.readFileSync(path.join(root, name), 'utf8');
  assert(html.indexOf('ceo@acitygames.com') === -1, name + ' must not contain ceo@acitygames.com');
});

var pilotSeq = fs.readFileSync(path.join(root, 'docs/PILOT_SEQUENCE_2026_2028.md'), 'utf8');
assert(pilotSeq.indexOf('## What already ships') !== -1, 'PILOT_SEQUENCE must name What already ships');
assert(pilotSeq.indexOf('## What a named org can buy time for') !== -1, 'PILOT_SEQUENCE must name What a named org can buy time for');
assert(pilotSeq.indexOf('## What remains SURMISE') !== -1, 'PILOT_SEQUENCE must name What remains SURMISE');
assert((pilotSeq.match(/^## /gm) || []).length === 3, 'PILOT_SEQUENCE must have three sections only');
assert(pilotSeq.indexOf('employ') !== -1 && pilotSeq.indexOf('Wrap') !== -1, 'PILOT_SEQUENCE ships section must name employ and wrap');
assert(pilotSeq.indexOf('Inquiry form') !== -1, 'PILOT_SEQUENCE ships section must name inquiry form');
assert(pilotSeq.indexOf('Inspectable lattice') !== -1, 'PILOT_SEQUENCE ships section must name inspectable lattice');
assert(pilotSeq.indexOf('GATE_EVAL') !== -1, 'PILOT_SEQUENCE ships section must name GATE_EVAL');
assert(pilotSeq.indexOf('human-replied AG-SML pilot') !== -1, 'PILOT_SEQUENCE must name human-replied AG-SML pilot');
assert(pilotSeq.indexOf('No license key arrives in the email') !== -1, 'PILOT_SEQUENCE must refuse a key in email');
assert(pilotSeq.indexOf('RBE as present') !== -1, 'PILOT_SEQUENCE SURMISE must name RBE as present');
assert(pilotSeq.indexOf('UBI') !== -1, 'PILOT_SEQUENCE SURMISE must name UBI');
assert(pilotSeq.indexOf('Token fade') !== -1, 'PILOT_SEQUENCE SURMISE must name token fade');
assert(pilotSeq.indexOf('Logistics') !== -1, 'PILOT_SEQUENCE SURMISE must name logistics');
assert(pilotSeq.indexOf('Combined AGSi') !== -1, 'PILOT_SEQUENCE SURMISE must keep Combined AGSi');
assert(pilotSeq.indexOf('grief-gated contributor class, not a graded abundance price and not a national accounts system') !== -1,
  'PILOT_SEQUENCE NEVC mention must be grief-gated contributor class, not a graded abundance price and not a national accounts system');
assert(pilotSeq.indexOf('we will house the working class') === -1, 'PILOT_SEQUENCE must not promise to house the working class');
assert(pilotSeq.indexOf('Do not change the walk') === -1, 'PILOT_SEQUENCE must not print operator HOLD');
assert(pilotSeq.indexOf('fonts.googleapis') === -1, 'PILOT_SEQUENCE must not load fonts.googleapis');
assert(pilotSeq.indexOf('ceo@acitygames.com') === -1, 'PILOT_SEQUENCE must not print ceo@acitygames.com');
assert(pilotSeq.indexOf('Ra-Thor ships fusion') === -1, 'PILOT_SEQUENCE must not claim Ra-Thor ships fusion');
assert(pilotSeq.indexOf('Stripe') === -1, 'PILOT_SEQUENCE must not name Stripe');
assert(briefingHtml.indexOf('PILOT_SEQUENCE_2026_2028.md') !== -1, 'briefing.html must point at the 24-month sequence file');
assert(briefingHtml.indexOf('Twenty-four months, said honestly') !== -1, 'briefing.html must carry the sequence card');
assert(briefingHtml.indexOf('human-replied AG-SML pilot') !== -1, 'briefing.html card must name human-replied AG-SML pilot');
assert(briefingHtml.indexOf('Do not change the walk') === -1, 'briefing.html must not print operator HOLD');
assert(briefingHtml.indexOf('fonts.googleapis') === -1, 'briefing.html must not load fonts.googleapis');
assert(briefingHtml.indexOf('we will house the working class') === -1, 'briefing.html must not promise to house the working class');
assert(briefingHtml.indexOf('Ra-Thor ships fusion') === -1, 'briefing.html must not claim Ra-Thor ships fusion');
var familyLinkBlock = familyNav.slice(familyNav.indexOf('var LINKS = ['), familyNav.indexOf('];', familyNav.indexOf('var LINKS = [')) + 2);
assert((familyLinkBlock.match(/href:/g) || []).length === 10, 'family walk is nine destinations plus Pilot after Employ');
assert(familyLinkBlock.indexOf("{ href: '/pilot.html', label: 'Pilot' }") !== -1, 'family walk must include Pilot after PILOT-NAV-POLISH');
assert(familyLinkBlock.indexOf("{ href: '/briefing.html'") === -1, 'family walk must not grow a briefing tab');

var commercialBrief = fs.readFileSync(path.join(root, 'docs/PUBLIC_COMMERCIAL_BRIEF.md'), 'utf8');
assert(employMd.indexOf('PUBLIC_COMMERCIAL_BRIEF.md') !== -1, 'docs/EMPLOY.md must point at PUBLIC_COMMERCIAL_BRIEF.md');
assert(commercialBrief.indexOf('14.15.6') !== -1, 'PUBLIC_COMMERCIAL_BRIEF.md must name workspace 14.15.6');
assert(commercialBrief.indexOf('AG-SML v1.1') !== -1, 'PUBLIC_COMMERCIAL_BRIEF.md must name AG-SML v1.1');
assert(commercialBrief.indexOf('info@Rathor.ai') !== -1, 'PUBLIC_COMMERCIAL_BRIEF.md must name info@Rathor.ai');
assert(commercialBrief.indexOf('commercial-inquiry') !== -1, 'PUBLIC_COMMERCIAL_BRIEF.md must point at commercial inquiry');
assert(commercialBrief.indexOf('Layer 0') !== -1, 'PUBLIC_COMMERCIAL_BRIEF.md must name Layer 0');
assert(commercialBrief.indexOf('inspect') !== -1 && commercialBrief.indexOf('METR') !== -1, 'PUBLIC_COMMERCIAL_BRIEF.md must keep inspect / METR');
assert(commercialBrief.indexOf('Stripe') === -1, 'PUBLIC_COMMERCIAL_BRIEF.md must not name Stripe');
assert(commercialBrief.indexOf('xAI partner') === -1, 'PUBLIC_COMMERCIAL_BRIEF.md must not print xAI partner');
assert(commercialBrief.indexOf('ceo@acitygames.com') === -1, 'PUBLIC_COMMERCIAL_BRIEF.md must not print ceo@acitygames.com');
assert(commercialBrief.indexOf('14.18') === -1, 'PUBLIC_COMMERCIAL_BRIEF.md must not sell 14.18');

function extractHeadingCard(html, heading) {
  var needle = '>' + heading + '</h2>';
  var idx = html.indexOf(needle);
  assert(idx !== -1, 'missing heading card: ' + heading);
  var start = html.lastIndexOf('<div class="card-hover', idx);
  assert(start !== -1, 'Organizations card must sit in a card-hover div');
  var next = html.indexOf('<div class="card-hover', idx);
  return html.slice(start, next === -1 ? html.length : next);
}

var orgCard = extractHeadingCard(employHtml, 'Organizations');
assert(orgCard.indexOf('commercial inquiry') !== -1, 'Organizations card must contain commercial inquiry');
assert(orgCard.indexOf('info@Rathor.ai') !== -1, 'Organizations card must contain info@Rathor.ai');
assert(orgCard.indexOf('/contact.html#commercial-inquiry') !== -1, 'Organizations card must link the commercial inquiry form');
assert(orgCard.indexOf('Stripe') === -1, 'Organizations card must not name Stripe');
assert(orgCard.indexOf('checkout') === -1, 'Organizations card must not name checkout');
assert(orgCard.indexOf('14.18') === -1, 'Organizations card must not sell 14.18');
assert(orgCard.indexOf('ceo@acitygames.com') === -1, 'Organizations card must not print ceo@acitygames.com');
assert(orgCard.indexOf('xAI partner') === -1, 'Organizations card must not print xAI partner');
assert(!/\bRBE\b/.test(orgCard) || /design thesis/.test(orgCard), 'Organizations card must not state RBE as present fact');
assert(employHtml.indexOf('>A. What you are employing<') !== -1, 'employ.html must keep A');
assert(employHtml.indexOf('>G. Honest gaps<') !== -1, 'employ.html must keep G');
assert(employHtml.indexOf('>F. License<') !== -1, 'employ.html must keep F. License');
assert(briefingHtml.indexOf('>Use cases<') !== -1, 'briefing.html must keep Use cases');
assert(briefingHtml.indexOf('>Organizations<') !== -1, 'briefing.html must carry Organizations card');
assert(briefingHtml.indexOf('>Twenty-four months, said honestly<') !== -1, 'briefing.html must keep Twenty-four months after Organizations');
assert(briefingHtml.indexOf('>Organizations<') < briefingHtml.indexOf('>Twenty-four months, said honestly<'), 'briefing Organizations card must sit before Twenty-four months');
assert(briefingHtml.indexOf('work habits') !== -1, 'briefing Organizations card may name work habits');
assert(briefingHtml.indexOf('not a national program') !== -1, 'briefing Organizations card must refuse a national program');
assert(briefingMd.indexOf('## Organizations') !== -1, 'PUBLIC_EMPLOY_BRIEFING.md must keep an Organizations section');
assert(briefingMd.indexOf('commercial inquiry') !== -1, 'PUBLIC_EMPLOY_BRIEFING.md Organizations copy must name commercial inquiry');
assert(briefingHtml.indexOf('Stripe') === -1, 'briefing.html must not name Stripe');
assert(employHtml.indexOf('Stripe') === -1, 'employ.html must not name Stripe');

console.log('EMPLOY-1 employ-spine-lock checks passed');
