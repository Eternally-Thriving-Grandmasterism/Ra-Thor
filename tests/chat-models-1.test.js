/* CHAT-MODELS-1: vendored WebLLM, picker ids, cache keep-list, system prompt bytes. */
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
var vendorRel = 'js/vendor/web-llm/0.2.85/index.js';
var vendorPath = path.join(root, vendorRel);
var vendor = fs.readFileSync(vendorPath, 'utf8');
var license = read('js/vendor/web-llm/0.2.85/LICENSE');

assert(chat.indexOf('https://esm.run/@mlc-ai/web-llm') === -1, 'chat must not import esm.run web-llm');
assert(chat.indexOf('esm.run') === -1, 'chat must not reference esm.run');
assert(chat.indexOf("./vendor/web-llm/0.2.85/index.js") !== -1, 'chat must import the pinned path relative to js/chat.js');
assert(chat.indexOf('WEBLLM_VENDOR') !== -1, 'vendor path must be named');

var vendorBytes = fs.statSync(vendorPath).size;
assert(vendorBytes === 6586947, 'vendored ESM must stay the pinned 0.2.85 dist (' + vendorBytes + ' bytes)');
console.log('vendored @mlc-ai/web-llm@0.2.85 index.js bytes: ' + vendorBytes);
assert(license.indexOf('Apache License') !== -1, 'Apache-2.0 LICENSE must sit next to the ESM');
assert(license.indexOf('Version 2.0') !== -1, 'LICENSE must be Apache-2.0');
assert(!fs.existsSync(path.join(root, 'js/vendor/web-llm/0.2.85/NOTICE')), '0.2.85 publishes no NOTICE file');

function stripComments(src) {
  return src
    .replace(/\/\*[\s\S]*?\*\//g, '')
    .replace(/(^|[^:\\])\/\/.*$/gm, '$1');
}

var code = stripComments(vendor);
assert(!/\bimport\s*\(/.test(code), 'vendored ESM must not dynamically import');
assert(!/\bimport\s+/.test(code), 'vendored ESM must not statically import');
assert(!/\bfrom\s+['"]/.test(code), 'vendored ESM must not use a from specifier');
assert(!/\brequire\s*\(\s*['"]/.test(code), 'vendored ESM must not require a specifier');
assert(vendor.indexOf('hasModelInCache') !== -1, 'pinned build must export hasModelInCache');
assert(vendor.indexOf('deleteModelAllInfoInCache') !== -1, 'pinned build must export deleteModelAllInfoInCache');
assert(vendor.indexOf('prebuiltAppConfig') !== -1, 'pinned build must export prebuiltAppConfig');
['webllm/model', 'webllm/wasm', 'webllm/config'].forEach(function (name) {
  assert(vendor.indexOf('"' + name + '"') !== -1, 'pinned build must open cache ' + name);
});

assert(sw.indexOf("key.indexOf('webllm') === -1") !== -1, 'activate keep list must retain webllm caches');
assert(sw.indexOf("var LOCK = '20260924a';") !== -1, 'LOCK must stay 20260924a');
assert(sw.indexOf("var CACHE = 'rathor-core-' + LOCK;") !== -1, 'CACHE name must stay rathor-core- plus LOCK');
var precache = sw.slice(sw.indexOf('var PRECACHE'), sw.indexOf('self.addEventListener'));
assert(precache.indexOf('web-llm') === -1, 'PRECACHE must not list the vendored library');
assert(precache.indexOf('vendor') === -1, 'PRECACHE must not list js/vendor');

var curatedStart = chat.indexOf('const WEBLLM_CURATED');
var curatedEnd = chat.indexOf('];', curatedStart);
assert(curatedStart !== -1 && curatedEnd !== -1, 'picker curated list must exist');
var curated = chat.slice(curatedStart, curatedEnd);
assert(curated.indexOf('Qwen2.5-3B') === -1, 'picker must not include Qwen2.5-3B');
assert(chat.indexOf('Qwen2.5-3B') === -1, 'chat.js must not name Qwen2.5-3B');

var bases = [
  'SmolLM2-360M-Instruct',
  'Qwen2.5-0.5B-Instruct',
  'Llama-3.2-1B-Instruct',
  'Qwen2.5-1.5B-Instruct',
  'gemma-2-2b-it',
  'Llama-3.2-3B-Instruct',
  'Phi-3.5-mini-instruct'
];
var pos = -1;
bases.forEach(function (base) {
  var at = curated.indexOf("base: '" + base + "'");
  assert(at > pos, 'curated order must keep ' + base);
  pos = at;
});
assert(chat.indexOf("const WEBLLM_DEFAULT_BASE = 'Llama-3.2-1B-Instruct'") !== -1, 'default base stays Llama-3.2-1B-Instruct');
assert(chat.indexOf("let llmModelId = 'Llama-3.2-1B-Instruct-q4f16_1-MLC'") !== -1, 'initial id stays the current q4f16 1B model');

var pickerIds = curated.match(/'[A-Za-z0-9._-]+-MLC'/g).map(function (token) {
  return token.slice(1, -1);
});
var curatedEntries = [];
var entryRe = /base: '([^']+)'[\s\S]*?q4f16: '([^']+)'[\s\S]*?q4f32: '([^']+)'/g;
var entryMatch;
while ((entryMatch = entryRe.exec(curated))) {
  curatedEntries.push({ base: entryMatch[1], q4f16: entryMatch[2], q4f32: entryMatch[3] });
}
assert(curatedEntries.length === 7, 'curated list stays seven bases');
var capMatch = chat.match(/const PHONE_MAX_VRAM_MB = ([0-9.]+);/);
assert(capMatch, 'PHONE_MAX_VRAM_MB must be a named constant');
var phoneMaxVram = Number(capMatch[1]);
assert(pickerIds.length === 14, 'picker must name q4f16_1 and q4f32_1 for each curated model');
var configIds = {};
var idRe = /model_id:\s*"([^"]+)"/g;
var idMatch;
while ((idMatch = idRe.exec(vendor))) configIds[idMatch[1]] = true;
var vramById = {};
pickerIds.forEach(function (id) {
  assert(configIds[id], 'picker id missing from pinned prebuiltAppConfig: ' + id);
  var at = vendor.indexOf('model_id: "' + id + '"');
  var slice = vendor.slice(at, at + 500);
  var vram = slice.match(/vram_required_MB:\s*([0-9.]+)/);
  assert(vram, 'vram_required_MB missing for ' + id);
  vramById[id] = Number(vram[1]);
  assert(chat.indexOf(vram[1]) === -1, 'chat.js must not hand-type vram ' + vram[1] + ' for ' + id);
  console.log(id + ' vram_required_MB=' + vram[1]);
});

assert(chat.indexOf('vram_required_MB') !== -1, 'size must be read from vram_required_MB');
assert(chat.indexOf("'shader-f16'") !== -1, 'q4f32_1 choice must depend on shader-f16');
assert(chat.indexOf('webllmShaderF16 ? entry.q4f16 : entry.q4f32') !== -1, 'q4f32_1 is only the branch when shader-f16 is absent');
assert(chat.indexOf("const WEBLLM_MODEL_KEY = 'rathor-webllm-model-v1'") !== -1, 'model choice key must be rathor-webllm-model-v1');
assert(chat.indexOf('hasModelInCache') !== -1, 'cache badge must call hasModelInCache');
assert(chat.indexOf('deleteModelAllInfoInCache') !== -1, 'delete must call deleteModelAllInfoInCache');
assert(chat.indexOf('engine.unload') !== -1, 'switching or deleting must unload the engine');
assert(chat.indexOf('window.confirm') === -1, 'chat must not call window.confirm');
var rowFlow = chat.slice(chat.indexOf('/* chat-models-1-pure */'), chat.indexOf('async function generateWithLocalLLM'));
assert(rowFlow.indexOf('confirm(') === -1, 'model rows must not call confirm');
assert(chat.indexOf("const WEBLLM_DEFAULT_BASE = 'Llama-3.2-1B-Instruct'") !== -1, 'Gemma must not become the default');
assert(chat.indexOf('gemma-2-2b-it') !== -1, 'gemma stays in the curated list');

var promptMarker = 'const SYSTEM_PROMPT = `';
var promptAt = chat.indexOf(promptMarker);
assert(promptAt !== -1, 'SYSTEM_PROMPT must stay a template literal');
var promptEnd = chat.indexOf('`;', promptAt);
var prompt = chat.slice(promptAt + promptMarker.length, promptEnd);
var promptHash = crypto.createHash('sha256').update(prompt).digest('hex');
assert(promptHash === '1fc78b6d442e43494de3bb52adc5307eb3c7861476cd0d838af0c3fcd61def61', 'SYSTEM_PROMPT bytes changed: ' + promptHash);
assert(Buffer.byteLength(prompt, 'utf8') === 840, 'SYSTEM_PROMPT byte length changed');

assert(chat.indexOf("if (!navigator.gpu) return { supported: false, reason: 'WebGPU not available in this browser' };") !== -1, 'WebGPU gate must stay');
assert(chat.indexOf('/Android|iPhone|iPad|iPod|Mobile/i.test(ua)') !== -1, 'mobile UA block must stay');
assert(chat.indexOf('async function detectLocalLlmSupport()') !== -1, 'support probe must be async');
assert((chat.match(/await detectLocalLlmSupport\(\)/g) || []).length === 2, 'init and i18n refresh both await the probe');
var detectStart = chat.indexOf('async function detectLocalLlmSupport()');
var detectEnd = chat.indexOf('function addMessage(', detectStart);
assert(detectStart !== -1 && detectEnd > detectStart, 'detectLocalLlmSupport must be extractable');
var detectSrc = chat.slice(detectStart, detectEnd);
assert(detectSrc.indexOf('/Android|iPhone|iPad|iPod|Mobile/i.test(ua)') < detectSrc.indexOf('requestAdapter()'), 'requestAdapter runs only after the phone UA test');
assert(detectSrc.indexOf('requestAdapter()') < detectSrc.indexOf('Local LLM currently works best on desktop.'), 'a null adapter still returns the phone block text');
assert(detectSrc.indexOf("return { supported: true, reason: null };") > detectSrc.indexOf('if (!adapter)'), 'a non-null adapter leaves the phone block');
var bootAt = chat.indexOf('const cap = await detectLocalLlmSupport();');
var boot = chat.slice(bootAt, chat.indexOf('applyChatSurfaceDir();', bootAt));
assert(boot.indexOf("if (!llmSupported) updateLlmUI('unsupported', cap.reason);") !== -1, 'a blocked probe still prints its reason');
assert(boot.indexOf('initWebllmPicker()') > boot.indexOf('if (!llmSupported)'), 'a supported probe opens the picker');
assert(chat.indexOf("webllmPicker.classList.remove('hidden')") !== -1, 'picker init shows the list');
var initSrc = chat.slice(chat.indexOf('async function initWebllmPicker'), chat.indexOf('async function generateWithLocalLLM'));
assert(initSrc.indexOf('keepCuratedRowOnPhone(webllmPhonePath, rec.vram_required_MB, PHONE_MAX_VRAM_MB)') !== -1, 'phone path filters rows with PHONE_MAX_VRAM_MB');
assert(initSrc.indexOf('rememberedModelPlan(') !== -1, 'init plans the remembered model against the listed rows');
assert(initSrc.indexOf('if (plan.writeDefault)') !== -1, 'init writes the model key only when none is stored');
assert(initSrc.indexOf('deleteModelAllInfoInCache') === -1, 'opening the picker does not delete a model cache');
assert(initSrc.indexOf('removeItem') === -1, 'opening the picker does not clear the stored model key');
assert(chat.indexOf('webllmPhonePath = cap.phone === true') !== -1, 'the row cap runs only when an adapter lifts the phone block');
assert(html.indexOf('id="webllm-phone-may-not-run" class="hidden') !== -1, 'the may-not-run line starts hidden');
assert(html.indexOf('id="webllm-phone-storage" class="hidden') !== -1, 'the storage line starts hidden');
assert(chat.indexOf('function generateLocalResponse') !== -1, 'fast responder must stay');
var send = chat.slice(chat.indexOf('async function sendMessage'), chat.indexOf('function exportSession'));
assert(send.indexOf('if (backendEnabled)') !== -1, 'local server stays ahead of WebLLM');
assert(send.indexOf('if (llmReady && llmEngine)') !== -1, 'WebLLM stays optional');
assert(send.indexOf('generateLocalResponse(text)') !== -1, 'fast responder stays the default path');
assert(chat.indexOf("chatStr('chatStatusDefault')") !== -1, 'idle status stays the fast-responder pack string');
assert(chat.indexOf("chatStr('chatPathFast')") !== -1, 'path badge stays the fast responder pack string');

assert(html.indexOf('Lattice Chat v14.18.x — offline-first multi-session store on your device. Workspace 14.15.6. Optional Web Crypto passphrase (PBKDF2 + AES-GCM). Capable · Bounded · Corrigible. Zero collection.') !== -1, 'chat meta string must stay');
assert(html.indexOf('v14.18.x • workspace 14.15.6 • TOLC 8 • family bar shared') !== -1, 'chat subtitle must stay');
assert(html.indexOf('Third-party models under their own licenses. Not made by Ra-Thor. Not reviewed or endorsed by their authors.') !== -1, 'third-party line must be on the page');
assert(html.indexOf("The first download of each model comes from Hugging Face and GitHub (raw.githubusercontent.com, which serves the model's code file) and needs the network. After that it runs in this browser.") !== -1, 'download line must name Hugging Face and GitHub');
assert(chat.indexOf("The first download of each model comes from Hugging Face and GitHub (raw.githubusercontent.com, which serves the model's code file) and needs the network. After that it runs in this browser.") !== -1, 'download fallback must name Hugging Face and GitHub');
assert(html.indexOf('Any other model: Local Server (Ollama).') !== -1, 'other models must point at Local Server (Ollama)');
assert(html.indexOf('Built with Llama') === -1, 'Built with Llama is rendered from the selected model, not a page-wide claim');
assert(chat.indexOf('Built with Llama') !== -1, 'Llama models must show Built with Llama');
assert(chat.indexOf('Llama 3.2 Community License') !== -1, 'Llama models must link the community license');
assert(chat.indexOf('Acceptable Use Policy') !== -1, 'Llama models must link the acceptable use policy');
assert(chat.indexOf('Gemma Terms of Use') !== -1, 'gemma must link Gemma Terms of Use');
assert(chat.indexOf('Prohibited Use Policy') !== -1, 'gemma must link the prohibited use policy');
assert(chat.indexOf('Apache-2.0') !== -1, 'Apache models must link Apache-2.0');
assert(chat.indexOf('>MIT<') === -1, 'MIT is a license link label in script, not markup');
assert(chat.indexOf("{ text: 'MIT'") !== -1, 'Phi must link MIT');

['best', 'smart', 'fast', 'reasoning', 'Ra-Thor model', 'Ra-Thor offline AI'].forEach(function (word) {
  assert(html.indexOf('id="webllm-third-party"') !== -1, 'third-party line marker');
  var blockStart = html.indexOf('id="webllm-picker"');
  var blockEnd = html.indexOf('id="backend-settings"');
  var block = html.slice(blockStart, blockEnd);
  assert(block.toLowerCase().indexOf(word.toLowerCase()) === -1, 'picker copy must not say ' + word);
});

assert(chat.indexOf("chatLabel('chatWebllmThirdParty'") !== -1, 'third-party line falls back when the pack key is missing');
assert(chat.indexOf("chatLabel('chatWebllmFirstDownload'") !== -1, 'download line falls back when the pack key is missing');
assert(html.indexOf('Offline only is on. Models already on this device still work.') !== -1, 'offline-only note must be on the page');
assert(html.indexOf('This switch applies to Lattice Chat models only.') !== -1, 'net mode scope must be on the page');
assert(chat.indexOf('rathor-net-mode-v1') !== -1, 'net mode key must be rathor-net-mode-v1');
assert(chat.indexOf("Offline only: Lattice Chat won't download models or contact any server except your own machine.") !== -1, 'offline-only sentence must be exact');
assert(chat.indexOf('Not on this device') !== -1, 'absent badge text');
assert(chat.indexOf('Partly downloaded') !== -1, 'partial badge text');
assert(chat.indexOf('On this device · works offline') !== -1, 'ready badge text');
assert(chat.indexOf('Download needs a connection.') !== -1, 'offline connection note');
assert(chat.indexOf('caches.keys') !== -1, 'partial detection must read Cache API names');
assert(chat.indexOf("indexOf('webllm')") !== -1, 'partial detection must look at webllm caches');

var downloadFn = chat.slice(chat.indexOf('async function startWebllmDownload'), chat.indexOf('async function loadWebllmModel'));
assert(downloadFn.indexOf('webllmDownloadAllowed') !== -1, 'download function must consult the guard');
assert(downloadFn.indexOf('.reload(') === -1, 'download function must not reload before the guard returns');
assert(downloadFn.indexOf('if (!webllmDownloadAllowed') < downloadFn.indexOf('loadWebllmModel'), 'guard must run before load');
var connectFn = chat.slice(chat.indexOf('async function connectBackend'), chat.indexOf('function disconnectBackend'));
assert(connectFn.indexOf('localServerEndpointAllowed') !== -1 && connectFn.indexOf('localServerEndpointAllowed') < connectFn.indexOf('fetch('), 'Local Server must check loopback before fetch');

var pure = chat.slice(chat.indexOf('/* chat-models-1-pure */'), chat.indexOf('/* chat-models-1-pure-end */'));
var store = {};
var sandbox = {
  URL: URL,
  localStorage: {
    getItem: function (key) { return Object.prototype.hasOwnProperty.call(store, key) ? store[key] : null; },
    setItem: function (key, value) { store[key] = String(value); }
  }
};
vm.createContext(sandbox);
vm.runInContext(pure + '\nthis.api = { readNetMode: readNetMode, writeNetMode: writeNetMode, webllmRowTransition: webllmRowTransition, webllmDownloadAllowed: webllmDownloadAllowed, localServerEndpointAllowed: localServerEndpointAllowed, webllmBadgeLabel: webllmBadgeLabel, webllmActionLabel: webllmActionLabel, absoluteScriptUrl: absoluteScriptUrl, modelCacheUrlPrefix: modelCacheUrlPrefix, requestMatchesModel: requestMatchesModel, purgeOwnedModelEntries: purgeOwnedModelEntries, modelFilesRemain: modelFilesRemain, phaseAfterLocalDelete: phaseAfterLocalDelete, tensorManifestDownloadBytes: tensorManifestDownloadBytes, formatByteMegabytes: formatByteMegabytes, downloadConsentText: downloadConsentText, deleteConsentText: deleteConsentText, isWebllmOwnedCache: isWebllmOwnedCache, keepCuratedRowOnPhone: keepCuratedRowOnPhone, listedModelBase: listedModelBase, rememberedModelPlan: rememberedModelPlan, phoneCapNotes: phoneCapNotes };', sandbox);
var api = sandbox.api;
assert(api.readNetMode() === 'network-on', 'default net mode is network-on');
assert(api.writeNetMode('offline-only') === 'offline-only', 'write offline-only');
assert(api.readNetMode() === 'offline-only', 'net mode persists in localStorage');
assert(store['rathor-net-mode-v1'] === 'offline-only', 'persisted value is offline-only');
assert(api.writeNetMode('network-on') === 'network-on', 'write network-on');
assert(api.readNetMode() === 'network-on', 'network-on persists');

assert(api.webllmDownloadAllowed('offline-only', true) === false, 'offline only blocks download');
assert(api.webllmDownloadAllowed('network-on', false) === false, 'no connection blocks download');
assert(api.webllmDownloadAllowed('network-on', true) === true, 'network on allows download');
assert(api.localServerEndpointAllowed('offline-only', 'http://localhost:11434/v1') === true, 'localhost allowed offline');
assert(api.localServerEndpointAllowed('offline-only', 'http://127.0.0.1:11434/v1') === true, '127.0.0.1 allowed offline');
assert(api.localServerEndpointAllowed('offline-only', 'http://[::1]:11434/v1') === true, '[::1] allowed offline');
assert(api.localServerEndpointAllowed('offline-only', 'https://example.com/v1') === false, 'non-loopback refused offline');
assert(api.localServerEndpointAllowed('network-on', 'https://example.com/v1') === true, 'network on keeps any Local Server endpoint');

var absent = { phase: 'absent', consent: null };
var tap1 = api.webllmRowTransition(absent, 'tap-download', { offlineOnly: false, onLine: true });
assert(tap1.effect === null, 'tap 1 must not download');
assert(tap1.row.consent === 'download', 'tap 1 shows download consent');
assert(tap1.row.phase === 'absent', 'tap 1 leaves the model off the device');
var blockedTap = api.webllmRowTransition(absent, 'tap-download', { offlineOnly: true, onLine: true });
assert(blockedTap.row.consent === null && blockedTap.effect === null, 'offline only does not open download consent');
var tap2 = api.webllmRowTransition(tap1.row, 'confirm-download', { offlineOnly: false, onLine: true });
assert(tap2.effect === 'download', 'tap 2 downloads');
assert(tap2.row.phase === 'downloading', 'tap 2 moves the row to downloading');
assert(tap2.row.consent === null, 'tap 2 closes consent');
var stopped = api.webllmRowTransition(tap2.row, 'stop', {});
assert(stopped.effect === 'stop' && stopped.row.phase === 'partial', 'stop leaves partly downloaded');
var delTap = api.webllmRowTransition({ phase: 'ready', consent: null }, 'tap-delete', {});
assert(delTap.effect === null && delTap.row.consent === 'delete', 'delete tap asks before removing');
assert(delTap.row.phase === 'ready', 'delete tap does not remove yet');
var delGone = api.webllmRowTransition({ phase: 'ready', consent: null }, 'confirm-delete', {});
assert(delGone.effect === null, 'confirm delete without the delete consent does nothing');
var delOk = api.webllmRowTransition(delTap.row, 'confirm-delete', {});
assert(delOk.effect === 'delete' && delOk.row.phase === 'absent', 'confirm delete clears the row');
var partialDel = api.webllmRowTransition({ phase: 'partial', consent: null }, 'confirm-delete', {});
assert(partialDel.effect === null, 'partial delete also needs its own confirm');
assert(api.webllmBadgeLabel('absent') === 'Not on this device');
assert(api.webllmBadgeLabel('partial') === 'Partly downloaded');
assert(api.webllmBadgeLabel('ready') === 'On this device · works offline');
assert(api.webllmBadgeLabel('downloading', 40) === 'Downloading 40%');
assert(api.webllmActionLabel('absent') === 'Download');
assert(api.webllmActionLabel('downloading') === 'Stop');
assert(api.webllmActionLabel('partial') === 'Delete');
assert(api.webllmActionLabel('ready') === 'Delete');

var modelId = 'SmolLM2-360M-Instruct-q4f32_1-MLC';
var modelUrl = 'https://huggingface.co/mlc-ai/' + modelId;
var siblingUrl = 'https://huggingface.co/mlc-ai/' + modelId + '-1k';
var prefix = api.modelCacheUrlPrefix(modelUrl);
var siblingPrefix = api.modelCacheUrlPrefix(siblingUrl);
var leftover = prefix + 'tensor-cache.json';
var siblingEntry = siblingPrefix + 'tensor-cache.json';
assert(siblingEntry.indexOf(modelId) !== -1, 'a prefix sibling URL still contains the shorter model id');
assert(api.requestMatchesModel(siblingEntry, modelUrl, null) === false, 'URL boundary must not match a prefix sibling');
assert(api.requestMatchesModel(leftover, modelUrl, null) === true, 'tensor-cache.json under the model URL is that model');
var stores = {
  'webllm/model': [leftover, siblingEntry],
  'webllm/script': [leftover],
  'rathor-core-20260924a': [leftover]
};
var phase = api.phaseAfterLocalDelete(stores, modelUrl, null);
assert(phase === 'absent', 'delete clears the leftover tensor-cache.json');
assert(api.webllmBadgeLabel(phase) === 'Not on this device', 'delete badge is Not on this device');
var after = api.purgeOwnedModelEntries(stores, modelUrl, null);
assert(after['webllm/model'].indexOf(leftover) === -1, 'owned cache drops this model');
assert(after['webllm/model'].indexOf(siblingEntry) !== -1, 'deleting one id leaves the prefix sibling');
assert(after['webllm/script'][0] === leftover, 'the script cache is not a model cache');
assert(after['rathor-core-20260924a'][0] === leftover, 'a cache this page does not own stays untouched');
assert(api.modelFilesRemain(after, modelUrl, null) === false, 'no owned entry remains for the deleted model');
assert(api.isWebllmOwnedCache('webllm/model') === true, 'webllm/model is owned');
assert(api.isWebllmOwnedCache('webllm/script') === false, 'the script cache is not swept as model data');
assert(api.isWebllmOwnedCache('tvmjs') === false, 'tvmjs is not this page or web-llm model cache');

var manifestBytes = api.tensorManifestDownloadBytes({
  records: [{ dataPath: 'params_shard_0.bin', nbytes: 203614080, records: [{ nbytes: 1 }] }]
});
assert(manifestBytes === 203614080, 'download bytes sum the manifest shard records');
assert(api.formatByteMegabytes(manifestBytes) === '203.6 MB', '203614080 bytes is 203.6 MB');
assert(api.downloadConsentText('SmolLM2-360M-Instruct', null).indexOf('Download size') === -1, 'no download size without a manifest sum');
assert(api.downloadConsentText('SmolLM2-360M-Instruct', manifestBytes).indexOf('Download size: 203.6 MB') !== -1, 'confirm shows the summed size');
assert(api.downloadConsentText('SmolLM2-360M-Instruct', null).indexOf('Hugging Face and GitHub') !== -1, 'confirm names both hosts');
assert(api.deleteConsentText('SmolLM2-360M-Instruct', null).indexOf('frees about') === -1, 'delete omits a size it did not sum');
assert(api.deleteConsentText('SmolLM2-360M-Instruct', manifestBytes).indexOf('This frees about 203.6 MB') !== -1, 'delete can quote a summed size');
assert(chat.indexOf('203.6') === -1, 'chat.js must not hand-type the SmolLM2 download size');
assert(chat.indexOf('GPU memory needed: about ') !== -1, 'vram is labeled as GPU memory');
assert(chat.indexOf('Small models can give wrong or inappropriate replies.') !== -1, 'smallest model row carries the reply note');

assert(chat.indexOf("const WEBLLM_SCRIPT_CACHE = 'webllm/script'") !== -1, 'vendor script uses the webllm/script cache');
assert(api.absoluteScriptUrl('https://rathor.ai/js/chat.js?v=20260924a', './vendor/web-llm/0.2.85/index.js') === 'https://rathor.ai/js/vendor/web-llm/0.2.85/index.js', 'script URL is absolute and has no query');
function activateWouldDelete(key) {
  var LOCK = '20260924a';
  return key.indexOf(LOCK) === -1 && key.indexOf('rathor-models') === -1 && key.indexOf('rathor-queue') === -1 && key.indexOf('webllm') === -1;
}
assert(activateWouldDelete('webllm/script') === false, 'webllm/script survives the activate keep list');
assert(sw.indexOf('req.destination === \'script\'') !== -1, 'script requests have a fetch fallback');
var scriptHandler = sw.slice(sw.indexOf("req.destination === 'script'"), sw.indexOf('event.respondWith(', sw.indexOf("req.destination === 'script'") + 80));
assert(scriptHandler.indexOf('caches.match(req)') !== -1, 'script fallback uses caches.match across caches');
assert(chat.indexOf('navigator.serviceWorker.ready') !== -1, 'download waits for the service worker when one is registered');
assert(chat.indexOf('cache.add(WEBLLM_SCRIPT_URL)') !== -1, 'download stores the vendored script');
var downloadFn2 = chat.slice(chat.indexOf('async function startWebllmDownload'), chat.indexOf('async function loadWebllmModel'));
assert(downloadFn2.indexOf('cacheVendoredWebllmScript') !== -1 && downloadFn2.indexOf('cacheVendoredWebllmScript') < downloadFn2.indexOf('loadWebllmModel'), 'script cache runs when a download starts');

var deleteFn = chat.slice(chat.indexOf('async function deleteWebllmModel'), chat.indexOf('async function useWebllmModel'));
assert(deleteFn.indexOf("readNetMode() !== 'offline-only'") !== -1, 'offline-only delete skips the network helper');
assert(deleteFn.indexOf("readNetMode() !== 'offline-only'") < deleteFn.indexOf('deleteModelAllInfoInCache'), 'the network delete is inside the online branch');
assert(deleteFn.indexOf('purgeModelLeftovers') !== -1, 'delete removes leftover cache entries');
assert(deleteFn.indexOf('fetch(') === -1, 'delete does not fetch');
assert(chat.indexOf('stopWebllmDownload') !== -1 && chat.indexOf('Download cancelled.') !== -1, 'offline only stops an in-progress download and says so');
assert(chat.indexOf('webllmPicker') !== -1 && chat.indexOf('scrollIntoView') !== -1, 'the WebLLM button focuses the picker');

assert(chat.indexOf('https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct/blob/a10cc1512eabd3dde888204e902eca88bddb4951/README.md') !== -1, 'SmolLM2 license link is a pinned README');
assert(chat.indexOf('https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/7ae557604adf67be50417f59c2c2f167def9a775/LICENSE') !== -1, 'Qwen 0.5B license is pinned');
assert(chat.indexOf('https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/LICENSE') !== -1, 'Qwen 1.5B license is pinned');
assert(chat.indexOf('https://github.com/meta-llama/llama-models/blob/8d29d93fa5700a60532e0061a02ffa89d0acd3fc/models/llama3_2/LICENSE') !== -1, 'Llama license is pinned');
assert(chat.indexOf('https://github.com/meta-llama/llama-models/blob/8d29d93fa5700a60532e0061a02ffa89d0acd3fc/models/llama3_2/USE_POLICY.md') !== -1, 'Llama use policy is pinned');
assert(chat.indexOf('https://huggingface.co/microsoft/Phi-3.5-mini-instruct/blob/2fe192450127e6a83f7441aef6e3ca586c338b77/LICENSE') !== -1, 'Phi license is pinned');
assert(chat.indexOf('https://ai.google.dev/gemma/terms') !== -1, 'Gemma terms stay on the page that has no revision');
assert(chat.indexOf('/blob/main/') === -1, 'license links must not float on blob/main');
assert(chat.indexOf('/resolve/main/LICENSE') === -1, 'license links must use blob, not resolve');

var en = read('i18n/en.js');
assert(en.indexOf('pick a model in the Local Intelligence list') !== -1, 'English greeting points at the model list');
assert(en.indexOf('enable WebLLM on desktop') === -1, 'English greeting must not say desktop-only');
assert(en.indexOf('"chatPhoneMayNotRun": "This model may not run on this device. Copy Context works everywhere."') !== -1, 'English pack has the may-not-run line');
assert(en.indexOf('"chatPhoneStorageEvict": "Safari may evict downloaded models when storage is low."') !== -1, 'English pack has the storage line');

var PHONE_BLOCK = 'Local LLM currently works best on desktop.';
var WEBGPU_BLOCK = 'WebGPU not available in this browser';
var iphone = 'Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Mobile/15E148 Safari/604.1';
var android = 'Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36';
var ipad = 'Mozilla/5.0 (iPad; CPU OS 17_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Mobile/15E148 Safari/604.1';
var desktop = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36';
var macDesktop = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15';
var SAFARI_EVICT = 'Safari may evict downloaded models when storage is low.';

var PHONE_BASES = ['SmolLM2-360M-Instruct', 'Qwen2.5-0.5B-Instruct', 'Llama-3.2-1B-Instruct'];
function rowsFor(shaderF16, phonePath) {
  var rows = [];
  curatedEntries.forEach(function (entry) {
    var id = shaderF16 ? entry.q4f16 : entry.q4f32;
    if (!api.keepCuratedRowOnPhone(phonePath, vramById[id], phoneMaxVram)) return;
    rows.push(entry.base);
  });
  return rows;
}

function probeSupport(nav) {
  var box = { navigator: nav };
  vm.createContext(box);
  vm.runInContext(detectSrc + '\nthis.go = detectLocalLlmSupport;', box);
  return box.go();
}

(async function () {
  var phoneCalls = 0;
  var adapterArgCount = -1;
  var lifted = await probeSupport({
    userAgent: iphone,
    gpu: {
      requestAdapter: function () {
        phoneCalls += 1;
        adapterArgCount = arguments.length;
        return Promise.resolve({ name: 'mock-adapter' });
      }
    }
  });
  assert(phoneCalls === 1, 'phone UA with navigator.gpu asks for an adapter once');
  assert(adapterArgCount === 0, 'requestAdapter is called with no arguments');
  assert(lifted.supported === true, 'phone UA with a non-null adapter is not UA-blocked');
  assert(lifted.reason === null, 'phone UA with a non-null adapter has no block text');
  assert(lifted.phone === true, 'a non-null adapter marks the phone path');
  curatedEntries.forEach(function (entry) {
    var under = PHONE_BASES.indexOf(entry.base) !== -1;
    [entry.q4f16, entry.q4f32].forEach(function (id) {
      var vram = vramById[id];
      if (under) assert(vram <= phoneMaxVram, id + ' stays at or under PHONE_MAX_VRAM_MB (' + vram + ')');
      else assert(vram > phoneMaxVram, id + ' stays above PHONE_MAX_VRAM_MB (' + vram + ')');
    });
  });
  var phoneF16 = rowsFor(true, true);
  var phoneF32 = rowsFor(false, true);
  assert(phoneF16.join('|') === PHONE_BASES.join('|'), 'shader-f16 phone picker lists the three smallest bases: ' + phoneF16.join(', '));
  assert(phoneF32.join('|') === PHONE_BASES.join('|'), 'q4f32 phone picker lists the three smallest bases: ' + phoneF32.join(', '));
  var notes = api.phoneCapNotes(true, iphone, 5);
  assert(notes.length === 2 && notes[0].indexOf('Copy Context works everywhere.') !== -1, 'iPhone shows the may-not-run line');
  assert(notes[1] === SAFARI_EVICT, 'iPhone shows the Safari storage line');
  assert(notes[0].toLowerCase().indexOf('supported') === -1, 'the may-not-run line does not say supported');
  var ipadNotes = api.phoneCapNotes(true, ipad, 5);
  assert(ipadNotes.length === 2 && ipadNotes[1] === SAFARI_EVICT, 'iPad shows the Safari storage line');
  var macTouchNotes = api.phoneCapNotes(true, macDesktop, 5);
  assert(macTouchNotes.length === 2 && macTouchNotes[1] === SAFARI_EVICT, 'Macintosh with touch points shows the Safari storage line');
  var macPlainNotes = api.phoneCapNotes(true, macDesktop, 0);
  assert(macPlainNotes.length === 1 && macPlainNotes[0].indexOf('may not run on this device') !== -1, 'Macintosh without touch points keeps the may-not-run line');
  assert(macPlainNotes.join('\n').indexOf(SAFARI_EVICT) === -1, 'Macintosh without touch points omits the Safari storage line');
  var androidNotes = api.phoneCapNotes(true, android, 5);
  assert(androidNotes.length === 1 && androidNotes[0].indexOf('may not run on this device') !== -1, 'Android with an adapter shows only the may-not-run line');
  assert(androidNotes.join('\n').indexOf(SAFARI_EVICT) === -1, 'Android does not show the Safari storage line');
  assert(chat.indexOf('phoneCapNotes(webllmPhonePath, phoneUa, phoneTouchPoints)') !== -1, 'the picker passes the user agent and touch points into the storage line');
  assert(chat.indexOf('navigator.maxTouchPoints') !== -1, 'Macintosh touch detection reads maxTouchPoints');
  var overCap = api.rememberedModelPlan('gemma-2-2b-it', phoneF16, 'Llama-3.2-1B-Instruct');
  assert(overCap.active === 'Llama-3.2-1B-Instruct', 'a remembered model above the cap uses the default on a phone');
  assert(overCap.stored === 'gemma-2-2b-it' && overCap.writeDefault === false, 'a remembered model above the cap keeps its stored key');
  var underCap = api.rememberedModelPlan('SmolLM2-360M-Instruct', phoneF16, 'Llama-3.2-1B-Instruct');
  assert(underCap.active === 'SmolLM2-360M-Instruct' && underCap.writeDefault === false, 'a remembered model under the cap stays selected');
  var emptyPlan = api.rememberedModelPlan('', phoneF16, 'Llama-3.2-1B-Instruct');
  assert(emptyPlan.active === '' && emptyPlan.stored === '' && emptyPlan.writeDefault === false, 'an empty stored key highlights no row and is not rewritten');
  var tapAgain = api.webllmRowTransition({ phase: 'absent', consent: null }, 'tap-download', { offlineOnly: false, onLine: true });
  var confirmAgain = api.webllmRowTransition(tapAgain.row, 'confirm-download', { offlineOnly: false, onLine: true });
  assert(tapAgain.effect === null && tapAgain.row.consent === 'download', 'two-tap download still asks on the first tap');
  assert(confirmAgain.effect === 'download' && confirmAgain.row.phase === 'downloading', 'two-tap confirm still downloads');

  var androidLift = await probeSupport({
    userAgent: android,
    gpu: { requestAdapter: function () { return Promise.resolve({ name: 'mock-adapter' }); } }
  });
  assert(androidLift.supported === true && androidLift.reason === null, 'Android UA with a non-null adapter is not UA-blocked');

  var noGpu = await probeSupport({ userAgent: iphone });
  assert(noGpu.supported === false, 'phone UA with no navigator.gpu stays blocked');
  assert(noGpu.reason === WEBGPU_BLOCK, 'phone UA with no navigator.gpu keeps the WebGPU block text');
  assert(noGpu.phone !== true, 'missing navigator.gpu does not open the phone picker');

  var nullAdapter = await probeSupport({
    userAgent: iphone,
    gpu: { requestAdapter: function () { return Promise.resolve(null); } }
  });
  assert(nullAdapter.supported === false, 'phone UA with a null adapter stays blocked');
  assert(nullAdapter.reason === PHONE_BLOCK, 'phone UA with a null adapter keeps the desktop block text');
  assert(nullAdapter.phone !== true, 'a null adapter does not open the phone picker');
  assert(api.phoneCapNotes(false).length === 0, 'a blocked phone shows neither phone line');

  var ipadNull = await probeSupport({
    userAgent: ipad,
    gpu: { requestAdapter: function () { return Promise.resolve(null); } }
  });
  assert(ipadNull.supported === false && ipadNull.reason === PHONE_BLOCK, 'iPad UA with a null adapter keeps the desktop block text');

  var rejected = await probeSupport({
    userAgent: android,
    gpu: { requestAdapter: function () { return Promise.reject(new Error('no adapter')); } }
  });
  assert(rejected.supported === false && rejected.reason === PHONE_BLOCK, 'a failed adapter request keeps the phone block text');

  var desktopCalls = 0;
  var desk = await probeSupport({
    userAgent: desktop,
    gpu: {
      requestAdapter: function () {
        desktopCalls += 1;
        return Promise.resolve(null);
      }
    }
  });
  assert(desktopCalls === 0, 'desktop probe does not call requestAdapter');
  assert(desk.supported === true && desk.reason === null, 'desktop with navigator.gpu stays available');
  assert(desk.phone !== true, 'desktop does not take the phone path');
  assert(rowsFor(true, false).length === 7 && rowsFor(false, false).length === 7, 'desktop lists all seven curated bases');
  assert(api.phoneCapNotes(false).length === 0, 'desktop shows neither phone line');

  var deskNoGpu = await probeSupport({ userAgent: desktop });
  assert(deskNoGpu.supported === false && deskNoGpu.reason === WEBGPU_BLOCK, 'desktop without WebGPU keeps the WebGPU block text');

  console.log('CHAT-MODELS-1 checks passed');
})().catch(function (err) {
  console.error(err);
  process.exit(1);
});
