/* CHAT-MODELS-1: vendored WebLLM, picker ids, cache keep-list, system prompt bytes. */
var fs = require('fs');
var path = require('path');
var crypto = require('crypto');

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
assert(chat.indexOf('chatWebllmDeleteConfirm') !== -1, 'delete must ask for confirmation');

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
assert(chat.indexOf("chatLabel('chatWebllmDownloaded'") !== -1, 'downloaded badge falls back when the pack key is missing');
assert(chat.indexOf("chatLabel('chatWebllmNotDownloaded'") !== -1, 'not-downloaded badge falls back when the pack key is missing');

console.log('CHAT-MODELS-1 checks passed');
