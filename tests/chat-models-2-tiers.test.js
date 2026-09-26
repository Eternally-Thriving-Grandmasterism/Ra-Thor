/* CHAT-MODELS-2: Light/Mid/Heavy tiers, Heavy gate, #570 follow-ups. */
var fs = require('fs');
var path = require('path');
var crypto = require('crypto');
var vm = require('vm');

var root = path.join(__dirname, '..');

function assert(cond, msg) {
  if (!cond) throw new Error(msg);
}

function read(rel) {
  return fs.readFileSync(path.join(root, rel), 'utf8');
}

var chat = read('js/chat.js');
var html = read('chat.html');
var sw = read('sw.js');
var en = read('i18n/en.js');
var vendor = read('js/vendor/web-llm/0.2.85/index.js');

assert(chat.indexOf("./vendor/web-llm/0.2.85/index.js") !== -1, 'WebLLM pin stays 0.2.85');
assert(chat.indexOf('0.2.86') === -1, 'this card must not bump the WebLLM pin');
assert(sw.indexOf("var LOCK = '20260924a';") !== -1, 'service worker LOCK stays 20260924a');
assert(chat.indexOf('DeepSeek') === -1, 'no DeepSeek id is pinned');
assert(chat.indexOf('<think>') === -1, 'no think-tag model is introduced');
assert(chat.indexOf('@huggingface/hub') === -1, 'no custom Hugging Face loader');
assert(chat.indexOf('window.confirm') === -1, 'no browser confirm dialog');

var promptMarker = 'const SYSTEM_PROMPT = `';
var promptAt = chat.indexOf(promptMarker);
var promptEnd = chat.indexOf('`;', promptAt);
var prompt = chat.slice(promptAt + promptMarker.length, promptEnd);
var promptHash = crypto.createHash('sha256').update(prompt).digest('hex');
assert(promptHash === '1fc78b6d442e43494de3bb52adc5307eb3c7861476cd0d838af0c3fcd61def61', 'SYSTEM_PROMPT bytes changed: ' + promptHash);

assert(html.indexOf('Lattice Chat v14.18.x — offline-first multi-session store on your device. Workspace 14.15.6. Optional Web Crypto passphrase (PBKDF2 + AES-GCM). Capable · Bounded · Corrigible. Zero collection.') !== -1, 'chat meta zero-collection line stays');
assert(html.indexOf('v14.18.x • workspace 14.15.6 • TOLC 8 • family bar shared') !== -1, 'chat subtitle stays');

var pure = chat.slice(chat.indexOf('/* chat-models-1-pure */'), chat.indexOf('/* chat-models-1-pure-end */'));
var store = {};
var sandbox = {
  URL: URL,
  localStorage: {
    getItem: function (key) { return Object.prototype.hasOwnProperty.call(store, key) ? store[key] : null; },
    setItem: function (key, value) { store[key] = String(value); },
    removeItem: function (key) { delete store[key]; }
  }
};
vm.createContext(sandbox);
vm.runInContext(pure + '\nthis.api = { webllmTierFromVram: webllmTierFromVram, heavyGateRequired: heavyGateRequired, heavyGateDecision: heavyGateDecision, storedModelAfterDelete: storedModelAfterDelete, headerLlmButtonState: headerLlmButtonState, webllmDownloadAllowed: webllmDownloadAllowed, webllmRowTransition: webllmRowTransition, downloadConsentText: downloadConsentText, formatStorageQuota: formatStorageQuota, storageTotalLabel: storageTotalLabel, adapterLimitsLabel: adapterLimitsLabel };', sandbox);
var api = sandbox.api;

assert(api.webllmTierFromVram(1199) === 'Light', 'just under 1200 MB is Light');
assert(api.webllmTierFromVram(1200) === 'Mid', '1200 MB starts Mid');
assert(api.webllmTierFromVram(2999) === 'Mid', 'just under 3000 MB stays Mid');
assert(api.webllmTierFromVram(3000) === 'Heavy', '3000 MB starts Heavy');
assert(api.heavyGateRequired('Light') === false, 'Light skips the Heavy gate');
assert(api.heavyGateRequired('Mid') === false, 'Mid skips the Heavy gate');
assert(api.heavyGateRequired('Heavy') === true, 'Heavy requires the gate');

var expectTier = {
  'SmolLM2-360M-Instruct-q4f16_1-MLC': 'Light',
  'SmolLM2-360M-Instruct-q4f32_1-MLC': 'Light',
  'Qwen2.5-0.5B-Instruct-q4f16_1-MLC': 'Light',
  'Qwen2.5-0.5B-Instruct-q4f32_1-MLC': 'Light',
  'Llama-3.2-1B-Instruct-q4f16_1-MLC': 'Light',
  'Llama-3.2-1B-Instruct-q4f32_1-MLC': 'Light',
  'Qwen2.5-1.5B-Instruct-q4f16_1-MLC': 'Mid',
  'Qwen2.5-1.5B-Instruct-q4f32_1-MLC': 'Mid',
  'gemma-2-2b-it-q4f16_1-MLC': 'Mid',
  'gemma-2-2b-it-q4f32_1-MLC': 'Mid',
  'Llama-3.2-3B-Instruct-q4f16_1-MLC': 'Mid',
  'Llama-3.2-3B-Instruct-q4f32_1-MLC': 'Mid',
  'Phi-3.5-mini-instruct-q4f16_1-MLC': 'Heavy',
  'Phi-3.5-mini-instruct-q4f32_1-MLC': 'Heavy'
};
Object.keys(expectTier).forEach(function (id) {
  var at = vendor.indexOf('model_id: "' + id + '"');
  assert(at !== -1, 'pinned id missing: ' + id);
  var slice = vendor.slice(at, at + 500);
  var vram = slice.match(/vram_required_MB:\s*([0-9.]+)/);
  assert(vram, 'vram missing for ' + id);
  var tier = api.webllmTierFromVram(Number(vram[1]));
  assert(tier === expectTier[id], id + ' vram ' + vram[1] + ' tier ' + tier + ' expected ' + expectTier[id]);
  console.log(id + ' tier=' + tier + ' vram_required_MB=' + vram[1]);
});

assert(chat.indexOf("return 'Light'") !== -1, 'Light label exists');
assert(chat.indexOf("return 'Mid'") !== -1, 'Mid label exists');
assert(chat.indexOf("return 'Heavy'") !== -1, 'Heavy label exists');
assert(chat.indexOf('webllm-tier') !== -1, 'rows are grouped under a tier heading');
assert(chat.indexOf('data-tier') !== -1, 'each row carries its tier');
assert(chat.indexOf('License:') !== -1, 'the row shows the license');
assert(chat.indexOf("const WEBLLM_DEFAULT_BASE = 'Llama-3.2-1B-Instruct'") !== -1, 'Gemma stays off the default');

var curatedStart = chat.indexOf('const WEBLLM_CURATED');
var curatedEnd = chat.indexOf('];', curatedStart);
var curated = chat.slice(curatedStart, curatedEnd);
var linkTexts = curated.match(/text: '([^']+)'/g).map(function (token) {
  return token.slice(7, -1);
});
var allowedLinks = {
  'Apache-2.0': true,
  'MIT': true,
  'Llama 3.2 Community License': true,
  'Acceptable Use Policy': true,
  'Gemma Terms of Use': true,
  'Prohibited Use Policy': true
};
linkTexts.forEach(function (text) {
  assert(allowedLinks[text], 'license label not on the allowed list: ' + text);
});
assert(linkTexts.indexOf('Apache-2.0') !== -1, 'Apache-2.0 is on a row');
assert(linkTexts.indexOf('MIT') !== -1, 'MIT is on a row');
assert(linkTexts.indexOf('Llama 3.2 Community License') !== -1, 'Llama community license is on a row');
assert(linkTexts.indexOf('Acceptable Use Policy') !== -1, 'Llama acceptable use policy is on a row');
assert(linkTexts.indexOf('Gemma Terms of Use') !== -1, 'Gemma terms are on a row');
assert(linkTexts.indexOf('Prohibited Use Policy') !== -1, 'Gemma prohibited use is on a row');
assert(chat.indexOf('Built with Llama') !== -1, 'Llama rows still say Built with Llama');

var heavyOk = api.heavyGateDecision({
  tier: 'Heavy',
  purpose: 'download',
  needBytes: 1000,
  limits: { maxBufferSize: 2147483648, maxStorageBufferBindingSize: 1073741824 },
  estimate: { quota: 8000000000, usage: 1000 }
});
assert(heavyOk.required === true && heavyOk.allow === true, 'Heavy proceeds when adapter limits and storage allow it');
assert(heavyOk.note.indexOf('GPU buffer limit:') !== -1, 'Heavy gate shows adapter limits');
assert(heavyOk.note.indexOf('Storage total:') !== -1, 'Heavy gate shows storage total');
assert(api.storageTotalLabel({ quota: 8000000000 }).indexOf('Storage total:') === 0, 'storage total label');

var heavyShort = api.heavyGateDecision({
  tier: 'Heavy',
  purpose: 'download',
  needBytes: 9000000000,
  limits: { maxBufferSize: 2147483648 },
  estimate: { quota: 5000000000, usage: 1000 }
});
assert(heavyShort.allow === false, 'Heavy download stops when storage is short');
assert(heavyShort.note.indexOf('Not enough storage for this download.') !== -1, 'short storage is stated');

var heavyUse = api.heavyGateDecision({
  tier: 'Heavy',
  purpose: 'use',
  needBytes: 9000000000,
  limits: { maxBufferSize: 2147483648 },
  estimate: { quota: 5000000000, usage: 1000 }
});
assert(heavyUse.allow === true, 'Heavy use of a model already chosen does not re-check download bytes');
assert(heavyUse.note.indexOf('Storage total:') !== -1, 'Heavy use still shows storage total');

var noAdapter = api.heavyGateDecision({
  tier: 'Heavy',
  purpose: 'download',
  limits: null,
  estimate: { quota: 8000000000, usage: 0 }
});
assert(noAdapter.allow === false, 'Heavy stops when adapter limits are missing');
assert(noAdapter.note.indexOf('GPU adapter limits are unavailable.') !== -1, 'missing adapter is stated');

var lightGate = api.heavyGateDecision({ tier: 'Light', purpose: 'download', limits: null, estimate: null, needBytes: 1 });
var midGate = api.heavyGateDecision({ tier: 'Mid', purpose: 'download', limits: null, estimate: null, needBytes: 1 });
assert(lightGate.required === false && lightGate.allow === true, 'Light does not require the Heavy gate');
assert(midGate.required === false && midGate.allow === true, 'Mid does not require the Heavy gate');

var actionFn = chat.slice(chat.indexOf('async function onWebllmRowAction'), chat.indexOf('async function initWebllmPicker'));
assert(actionFn.indexOf('heavyGateRequired') !== -1, 'row actions consult the Heavy gate');
assert(actionFn.indexOf('readHeavyGate') !== -1, 'row actions read adapter limits and storage');
assert(actionFn.indexOf('persistOriginStorage') !== -1, 'Heavy confirm can persist storage');
var persistFn = chat.slice(chat.indexOf('async function persistOriginStorage'), chat.indexOf('async function onWebllmRowAction'));
assert(persistFn.indexOf('navigator.storage.persist') !== -1, 'persist uses navigator.storage.persist where available');
var readFn = chat.slice(chat.indexOf('async function readHeavyGate'), chat.indexOf('async function persistOriginStorage'));
assert(readFn.indexOf('requestAdapter') !== -1, 'Heavy gate reads adapter limits');
assert(readFn.indexOf('navigator.storage.estimate') !== -1, 'Heavy gate reads storage estimate');
assert(actionFn.indexOf("if (heavy)") !== -1, 'persist and the gate sit on the Heavy branch');
assert(actionFn.indexOf('startWebllmDownload') > actionFn.indexOf('persistOriginStorage'), 'persist runs before a Heavy download starts');

assert(api.webllmDownloadAllowed('offline-only', true) === false, 'offline only still blocks download');
var offlineTap = api.webllmRowTransition(
  { phase: 'absent', consent: null },
  'tap-download',
  { offlineOnly: true, onLine: true, heavy: true }
);
assert(offlineTap.effect === null && offlineTap.row.consent === null, 'offline only does not open Heavy download consent');
var offlineConfirm = api.webllmRowTransition(
  { phase: 'absent', consent: 'download' },
  'confirm-download',
  { offlineOnly: true, onLine: true, heavy: true, heavyBlocked: false }
);
assert(offlineConfirm.effect === null, 'offline only still blocks confirm-download');

var lightUse = api.webllmRowTransition({ phase: 'ready', consent: null }, 'use', { heavy: false, offlineOnly: false, onLine: true });
assert(lightUse.effect === 'use', 'Light and Mid use stays one tap');
var heavyTap = api.webllmRowTransition({ phase: 'ready', consent: null }, 'use', { heavy: true, offlineOnly: false, onLine: true });
assert(heavyTap.effect === null && heavyTap.row.consent === 'heavy-use', 'Heavy use asks before it runs');
var heavyBlocked = api.webllmRowTransition(
  { phase: 'absent', consent: 'download' },
  'confirm-download',
  { offlineOnly: false, onLine: true, heavy: true, heavyBlocked: true }
);
assert(heavyBlocked.effect === null && heavyBlocked.row.consent === 'download', 'a failed Heavy gate does not start the download');
var heavyAllowed = api.webllmRowTransition(
  { phase: 'absent', consent: 'download' },
  'confirm-download',
  { offlineOnly: false, onLine: true, heavy: true, heavyBlocked: false }
);
assert(heavyAllowed.effect === 'download', 'a passing Heavy gate still uses the confirm tap');

var sizeSentence = 'A small size file is fetched to show the download size.';
assert(html.indexOf(sizeSentence) !== -1, 'picker copy discloses the size-file fetch');
assert(chat.indexOf(sizeSentence) !== -1, 'chat discloses the size-file fetch');
assert(api.downloadConsentText('SmolLM2-360M-Instruct', null).indexOf(sizeSentence) !== -1, 'confirm copy discloses the size-file fetch');
var tapAt = actionFn.indexOf("action === 'tap-download'");
var fetchAt = actionFn.indexOf('fetchTensorDownloadBytes');
var effectAt = actionFn.indexOf("result.effect === 'download'");
assert(tapAt !== -1 && fetchAt > tapAt && fetchAt < effectAt, 'size file is still fetched on Download, before Confirm, so the confirm line can show the size');

var fallbackMatch = en.match(/"chatReplyFallback": "([\s\S]*?)",?\s*\n\}/);
assert(fallbackMatch, 'chatReplyFallback must exist');
assert(fallbackMatch[1].indexOf('Local Intelligence') !== -1, 'fallback points at the Local Intelligence list');
assert(fallbackMatch[1].indexOf('supported desktops') === -1, 'fallback must not say supported desktops');
assert(en.indexOf('Enable **WebLLM** on supported desktops') === -1, 'English pack drops the desktop-only WebLLM line');

assert(api.storedModelAfterDelete('Phi-3.5-mini-instruct', 'Phi-3.5-mini-instruct') === '', 'deleting the stored model clears the key');
assert(api.storedModelAfterDelete('Llama-3.2-1B-Instruct', 'Phi-3.5-mini-instruct') === 'Llama-3.2-1B-Instruct', 'deleting another row keeps the stored model');
var deleteFn = chat.slice(chat.indexOf('async function deleteWebllmModel'), chat.indexOf('async function useWebllmModel'));
assert(deleteFn.indexOf('storedModelAfterDelete') !== -1, 'delete consults the stored-model helper');
assert(deleteFn.indexOf('removeItem(WEBLLM_MODEL_KEY)') !== -1, 'delete removes rathor-webllm-model-v1');
var renderFn = chat.slice(chat.indexOf('function renderWebllmRows'), chat.indexOf('async function unloadWebllmEngine'));
assert(renderFn.indexOf('readStoredModelBase()') !== -1, 'highlight follows the stored key');
assert(renderFn.indexOf('activeModelBase') === -1, 'highlight does not fall back onto a deleted default');

var hiddenIdle = api.headerLlmButtonState({ pickerPresent: true, pickerHidden: true, uiState: 'idle' });
assert(hiddenIdle.disabled === true && hiddenIdle.action === 'none', 'a hidden picker disables the header button');
assert(hiddenIdle.label === 'Models list unavailable', 'hidden picker names that state');
var hiddenError = api.headerLlmButtonState({ pickerPresent: true, pickerHidden: true, uiState: 'error' });
assert(hiddenError.disabled === false && hiddenError.action === 'retry', 'an error is a retry, not a dead click');
assert(hiddenError.label === 'Try again', 'error label stays Try again');
var visibleError = api.headerLlmButtonState({ pickerPresent: true, pickerHidden: false, uiState: 'error' });
assert(visibleError.action === 'retry', 'a visible picker in error still retries');
var missing = api.headerLlmButtonState({ pickerPresent: false, pickerHidden: true, uiState: 'idle' });
assert(missing.disabled === true && missing.action === 'none', 'a missing picker is not a live button');
var unsupported = api.headerLlmButtonState({ pickerPresent: true, pickerHidden: true, uiState: 'unsupported' });
assert(unsupported.disabled === true && unsupported.action === 'none', 'unsupported stays disabled');
var ready = api.headerLlmButtonState({ pickerPresent: true, pickerHidden: false, uiState: 'ready' });
assert(ready.disabled === false && ready.action === 'focus', 'a visible ready list can be focused');
var btnAt = html.indexOf('id="local-llm-btn"');
var btn = html.slice(btnAt - 40, btnAt + 420);
assert(btn.indexOf('disabled') !== -1, 'header button starts disabled while the picker is hidden');
assert(btn.indexOf('data-llm-action="none"') !== -1, 'header button starts with an explicit action');
assert(chat.indexOf('headerLlmButtonState') !== -1, 'click path uses the header state helper');
assert(chat.indexOf("header.action === 'retry'") !== -1, 'error click retries the list');
assert(chat.indexOf("header.action !== 'focus'") !== -1, 'a hidden list does not pretend to focus');

console.log('CHAT-MODELS-2 tier checks passed');
