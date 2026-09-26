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
assert(pickerIds.length === 14, 'picker must name q4f16_1 and q4f32_1 for each curated model');
var configIds = {};
var idRe = /model_id:\s*"([^"]+)"/g;
var idMatch;
while ((idMatch = idRe.exec(vendor))) configIds[idMatch[1]] = true;
pickerIds.forEach(function (id) {
  assert(configIds[id], 'picker id missing from pinned prebuiltAppConfig: ' + id);
  var at = vendor.indexOf('model_id: "' + id + '"');
  var slice = vendor.slice(at, at + 500);
  var vram = slice.match(/vram_required_MB:\s*([0-9.]+)/);
  assert(vram, 'vram_required_MB missing for ' + id);
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
assert(html.indexOf('The first download of each model comes from Hugging Face and needs the network. After that it runs in this browser.') !== -1, 'download line must be on the page');
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
vm.runInContext(pure + '\nthis.api = { readNetMode: readNetMode, writeNetMode: writeNetMode, webllmRowTransition: webllmRowTransition, webllmDownloadAllowed: webllmDownloadAllowed, localServerEndpointAllowed: localServerEndpointAllowed, webllmBadgeLabel: webllmBadgeLabel, webllmActionLabel: webllmActionLabel };', sandbox);
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

console.log('CHAT-MODELS-1 checks passed');
