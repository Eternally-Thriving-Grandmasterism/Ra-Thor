/* PICKER-STEP-1: the phone lists only Qwen2.5-0.5B-Instruct; desktop gains the
 * Qwen2.5-7B-Instruct q4f16 row from the pinned WebLLM 0.2.85 prebuiltAppConfig.
 * Sizes come from the pin (vram_required_MB) and from the model's own tensor-cache.json
 * at Download time. Nothing is hand-typed. Contact: info@Rathor.ai */
var fs = require('fs');
var path = require('path');
var vm = require('vm');
var root = path.join(__dirname, '..');

function assert(cond, msg) {
  if (!cond) throw new Error(msg);
}
function read(rel) {
  return fs.readFileSync(path.join(root, rel), 'utf8');
}

var chat = read('js/chat.js');
var sw = read('sw.js');
var vendor = read('js/vendor/web-llm/0.2.85/index.js');
var NEW_ID = 'Qwen2.5-7B-Instruct-q4f16_1-MLC';
var LABEL = 'Desktop. Not SuperGrok. Needs a real GPU.';

// 1. The pin has the id; the size fields come from the pin's own record.
var at = vendor.indexOf('model_id: "' + NEW_ID + '"');
assert(at !== -1, NEW_ID + ' must be in the pinned 0.2.85 prebuiltAppConfig');
var rec = vendor.slice(at, vendor.indexOf('model_id:', at + 10));
var vram = rec.match(/vram_required_MB:\s*([0-9.]+)/);
assert(vram, 'pinned record carries vram_required_MB');
assert(chat.indexOf(vram[1]) === -1, 'chat.js must not hand-type the pinned vram ' + vram[1]);
assert(!/download_size|size_bytes|model_size/.test(vendor.slice(at, at + 500)), 'the pinned record has no download-size field; the size is read from tensor-cache.json');
assert(chat.indexOf('4284') === -1, 'chat.js must not hand-type a download size');
assert(chat.indexOf("./vendor/web-llm/0.2.85/index.js") !== -1 && chat.indexOf('0.2.86') === -1, 'WebLLM pin stays 0.2.85');

// 2. Curated entry: q4f16 only, exact label, desktop only.
var curated = chat.slice(chat.indexOf('const WEBLLM_CURATED'), chat.indexOf('];', chat.indexOf('const WEBLLM_CURATED')));
var entryAt = curated.indexOf("base: 'Qwen2.5-7B-Instruct'");
assert(entryAt !== -1, 'curated list has Qwen2.5-7B-Instruct');
var entry = curated.slice(entryAt);
assert(entry.indexOf("q4f16: '" + NEW_ID + "'") !== -1, 'the new row uses the pinned q4f16 id');
assert(entry.indexOf('q4f32: null') !== -1, 'no q4f32 id is added for the new row');
assert(entry.indexOf("desktopLabel: '" + LABEL + "'") !== -1, 'the new row label is exact');
assert(entry.indexOf('phoneRow') === -1, 'the new row is not a phone row');
assert(curated.indexOf('Llama-3.1-8B') === -1, 'Llama-3.1-8B is not added when Qwen2.5-7B is in the pin');
var render = chat.slice(chat.indexOf('function renderWebllmRows'), chat.indexOf('async function unloadWebllmEngine'));
assert(render.indexOf('desk.textContent = opt.entry.desktopLabel;') !== -1, 'the row renders its desktop label');

// 3. Phone list is exactly Qwen2.5-0.5B; cap stays; stored key is never rewritten.
var phoneRows = (curated.match(/phoneRow: true/g) || []).length;
assert(phoneRows === 1, 'exactly one curated row is marked for the phone');
var qwenAt = curated.indexOf("base: 'Qwen2.5-0.5B-Instruct'");
var qwenNext = curated.indexOf("base: '", qwenAt + 1);
assert(curated.slice(qwenAt, qwenNext).indexOf('phoneRow: true') !== -1, 'the phone row is Qwen2.5-0.5B-Instruct');
['SmolLM2-360M-Instruct', 'Llama-3.2-1B-Instruct', 'Qwen2.5-0.5B-Instruct'].forEach(function (base) {
  assert(curated.indexOf("base: '" + base + "'") !== -1, base + ' stays in WEBLLM_CURATED for desktop');
});
assert(chat.indexOf('const PHONE_MAX_VRAM_MB = 1200;') !== -1, 'PHONE_MAX_VRAM_MB stays 1200');
var init = chat.slice(chat.indexOf('async function initWebllmPicker'), chat.indexOf('async function generateWithLocalLLM'));
assert(init.indexOf('entry.phoneRow === true') !== -1, 'init passes the phone flag');
assert(init.indexOf('removeItem') === -1, 'init does not clear the stored key');
assert((init.match(/localStorage\.setItem\(WEBLLM_MODEL_KEY/g) || []).length === 1 && init.indexOf('if (plan.writeDefault)') !== -1, 'init writes the key only behind writeDefault');

// 4. Pure helpers: phone filter, highlight, consent text, two-tap, Heavy tier.
var pure = chat.slice(chat.indexOf('/* chat-models-1-pure */'), chat.indexOf('/* chat-models-1-pure-end */'));
var sandbox = { URL: URL, localStorage: { getItem: function () { return null; }, setItem: function () {}, removeItem: function () {} } };
vm.createContext(sandbox);
vm.runInContext(pure + '\nthis.api = { keepCuratedRowOnPhone: keepCuratedRowOnPhone, phoneHighlightPlan: phoneHighlightPlan, rememberedModelPlan: rememberedModelPlan, downloadConsentText: downloadConsentText, webllmRowTransition: webllmRowTransition, webllmTierFromVram: webllmTierFromVram, heavyGateRequired: heavyGateRequired };', sandbox);
var api = sandbox.api;
assert(api.keepCuratedRowOnPhone(true, 376, 1200, true) === true, 'a marked row under the cap is on the phone');
assert(api.keepCuratedRowOnPhone(true, 376, 1200, false) === false, 'an unmarked row under the cap is off the phone');
assert(api.keepCuratedRowOnPhone(false, Number(vram[1]), 1200, false) === true, 'desktop keeps the 7B row');
var plan = api.rememberedModelPlan('Llama-3.2-1B-Instruct', ['Qwen2.5-0.5B-Instruct'], 'Llama-3.2-1B-Instruct');
assert(plan.stored === 'Llama-3.2-1B-Instruct' && plan.writeDefault === false, 'a stored key is kept, not migrated');
var hl = api.phoneHighlightPlan(true, 'SmolLM2-360M-Instruct', ['Qwen2.5-0.5B-Instruct'], false, plan.active);
assert(hl.active === 'Qwen2.5-0.5B-Instruct' && hl.stored === 'SmolLM2-360M-Instruct' && hl.writeDefault === false, 'phone highlights Qwen and keeps the stored key');
var consent = api.downloadConsentText('Qwen2.5-7B-Instruct', 4284263424);
assert(consent.indexOf('Download size: 4284.3 MB.') !== -1, 'consent shows the real byte total it is given: ' + consent);
assert(api.downloadConsentText('Qwen2.5-7B-Instruct', null).indexOf('Download size') === -1, 'no size is shown when the size file is unavailable');
var tap = api.webllmRowTransition({ phase: 'absent', consent: null }, 'tap-download', { offlineOnly: false, onLine: true, heavy: true });
assert(tap.effect === null && tap.row.consent === 'download', 'first tap only asks');
var ok = api.webllmRowTransition(tap.row, 'confirm-download', { offlineOnly: false, onLine: true, heavy: true, heavyBlocked: false });
assert(ok.effect === 'download', 'second tap downloads');
var blocked = api.webllmRowTransition(tap.row, 'confirm-download', { offlineOnly: false, onLine: true, heavy: true, heavyBlocked: true });
assert(blocked.effect === null, 'the Heavy gate still blocks');
assert(api.webllmTierFromVram(Number(vram[1])) === 'Heavy' && api.heavyGateRequired('Heavy') === true, 'the 7B row is Heavy and goes through the Heavy gate');

// 5. Out-of-scope files stay put.
assert(sw.indexOf("var LOCK = '20261001b';") !== -1, 'sw.js LOCK stays 20261001b');

console.log('picker-step-1: phone rows 1 (Qwen2.5-0.5B); desktop adds ' + NEW_ID + ' (pin vram_required_MB ' + vram[1] + '); label "' + LABEL + '"');
