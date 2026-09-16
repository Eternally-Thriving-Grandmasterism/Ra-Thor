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
assert(indexHtml.indexOf('WHITEPAPER_v4.2') !== -1, 'index.html must point at living cover v4.2');
assert(indexHtml.indexOf('WHITEPAPER_v4.1') === -1, 'index.html must not present v4.1 as the living cover');
assert(indexHtml.indexOf('Whitepaper v4.1') === -1, 'index.html must not advertise v4.1 as the living cover');

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

console.log('EMPLOY-1 employ-spine-lock checks passed');
