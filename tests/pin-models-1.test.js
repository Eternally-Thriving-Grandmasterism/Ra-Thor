/* PIN-MODELS-1: pinned HF revisions and wasm git sha for the curated picker. */
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
var html = read('chat.html');
var sw = read('sw.js');
var vendor = read('js/vendor/web-llm/0.2.85/index.js');

assert(chat.indexOf('resolve/main') === -1, 'js/chat.js must not name a floating HF revision');
assert(chat.indexOf('binary-mlc-llm-libs/main') === -1, 'js/chat.js must not name a floating wasm ref');
assert(html.indexOf('resolve/main') === -1, 'chat.html must not name a floating HF revision');
assert(html.indexOf('binary-mlc-llm-libs/main') === -1, 'chat.html must not name a floating wasm ref');
assert(sw.indexOf('resolve/main') === -1, 'sw.js does not cache a floating HF revision');
assert(sw.indexOf('binary-mlc-llm-libs/main') === -1, 'sw.js does not cache a floating wasm ref');
assert(vendor.indexOf('binary-mlc-llm-libs/main') !== -1, 'the vendored library still carries its own floating wasm prefix');
assert(vendor.indexOf('resolve/main/') !== -1, 'the vendored cleanModelUrl still documents the floating revision');

var notice = read('js/vendor/web-llm/0.2.85/NOTICE').replace(/\n$/, '');
assert(notice === 'This product includes software developed at The Apache Software Foundation.', 'NOTICE line');
assert(read('js/vendor/web-llm/0.2.85/LICENSE').indexOf('Apache License') !== -1, 'LICENSE is untouched Apache-2.0');

var curatedStart = chat.indexOf('const WEBLLM_CURATED');
var curatedSrc = chat.slice(curatedStart, chat.indexOf('];', curatedStart) + 2);
assert(curatedSrc.indexOf('gemma') === -1, 'curated source has no gemma row');
var pinSrc = chat.slice(chat.indexOf('const WEBLLM_WASM_REVISION'), chat.indexOf('/* pin-models-1-pure-end */'));
var sandbox = { URL: URL };
vm.createContext(sandbox);
vm.runInContext(curatedSrc + '\n' + pinSrc + '\nthis.api = { pinModelRecord: pinModelRecord, pinWasmLibUrl: pinWasmLibUrl, curatedWebllmAppConfig: curatedWebllmAppConfig, WEBLLM_HF_REVISIONS: WEBLLM_HF_REVISIONS, WEBLLM_WASM_REVISION: WEBLLM_WASM_REVISION, WEBLLM_CURATED: WEBLLM_CURATED };', sandbox);
var api = sandbox.api;
assert(api.WEBLLM_WASM_REVISION === '146ed9cd4b4062e967b017fd2291f7106b50599f', 'wasm git sha');
assert(Object.keys(api.WEBLLM_HF_REVISIONS).length === 13, 'thirteen weight revisions');
assert(api.WEBLLM_CURATED.length === 7, 'seven picker rows');

function cleanModelUrl(modelUrl) {
  modelUrl += modelUrl.endsWith('/') ? '' : '/';
  if (!modelUrl.match(/.+\/resolve\/.+\//)) modelUrl += 'resolve/main/';
  return new URL(modelUrl).href;
}

function vendorRecord(id) {
  var at = vendor.indexOf('model_id: "' + id + '"');
  assert(at !== -1, 'vendored record missing ' + id);
  var modelAt = vendor.lastIndexOf('model: "', at);
  var model = vendor.slice(modelAt, at).match(/"([^"]+)"/);
  assert(model, 'vendored model URL missing ' + id);
  var slice = vendor.slice(at, at + 600);
  var file = slice.match(/"\/([^"]+\.wasm)"/);
  assert(file, 'vendored wasm name missing ' + id);
  var vram = slice.match(/vram_required_MB:\s*([0-9.]+)/);
  assert(vram, 'vendored vram missing ' + id);
  var features = slice.match(/required_features:\s*\[([^\]]*)\]/);
  var rec = {
    model: model[1],
    model_id: id,
    model_lib: 'https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/web-llm-models/v0_2_84/base/' + file[1],
    vram_required_MB: Number(vram[1]),
    overrides: { context_window_size: 4096 }
  };
  if (features && features[1].indexOf('shader-f16') !== -1) rec.required_features = ['shader-f16'];
  return rec;
}

var ids = [];
api.WEBLLM_CURATED.forEach(function (entry) {
  [entry.q4f16, entry.q4f32].forEach(function (id) {
    if (id) ids.push(id);
  });
});
assert(ids.length === 13, 'curated weight ids are thirteen');
assert(ids[4] === 'Llama-3.2-1B-Instruct-q4f16_1-MLC', 'default weight stays the fifth id, Llama 3.2 1B q4f16');
assert(ids.indexOf('gemma-2-2b-it-q4f16_1-MLC') === -1, 'gemma q4f16 is not pinned');
assert(ids.indexOf('gemma-2-2b-it-q4f32_1-MLC') === -1, 'gemma q4f32 is not pinned');

var modelList = ids.map(vendorRecord);
var config = api.curatedWebllmAppConfig({
  prebuiltAppConfig: { cacheBackend: 'cache', model_list: modelList }
});
assert(config.cacheBackend === 'cache', 'cache backend is copied from the vendored app config');
assert(config.model_list.length === 13, 'appConfig lists the thirteen curated weights');

var wasmSha = api.WEBLLM_WASM_REVISION;
config.model_list.forEach(function (rec, i) {
  var sha = api.WEBLLM_HF_REVISIONS[rec.model_id];
  assert(sha, 'sha for ' + rec.model_id);
  var expectModel = 'https://huggingface.co/mlc-ai/' + rec.model_id + '/resolve/' + sha + '/';
  assert(rec.model === expectModel, rec.model_id + ' model URL ' + rec.model);
  assert(cleanModelUrl(rec.model) === expectModel, 'cleanModelUrl keeps the pinned revision for ' + rec.model_id);
  assert(cleanModelUrl(rec.model).indexOf('resolve/main') === -1, 'cleanModelUrl must not float ' + rec.model_id);
  assert(rec.model_lib.indexOf('/binary-mlc-llm-libs/' + wasmSha + '/web-llm-models/v0_2_84/base/') !== -1, 'wasm dir for ' + rec.model_id);
  assert(rec.model_lib.indexOf('/main/') === -1, 'wasm ref is not main for ' + rec.model_id);
  assert(rec.model_lib.slice(rec.model_lib.lastIndexOf('/') + 1) === modelList[i].model_lib.slice(modelList[i].model_lib.lastIndexOf('/') + 1), 'wasm file name stays the vendored name for ' + rec.model_id);
  assert(rec.vram_required_MB === modelList[i].vram_required_MB, 'vram is copied for ' + rec.model_id);
  assert(rec.overrides.context_window_size === 4096, 'context override is copied for ' + rec.model_id);
  assert(rec.model_id === ids[i], 'appConfig order follows the picker for ' + rec.model_id);
  console.log(rec.model_id + ' ' + sha);
  console.log('  model ' + rec.model);
  console.log('  wasm  ' + rec.model_lib);
});

var qwen = config.model_list.filter(function (rec) {
  return rec.model_id === 'Qwen2.5-0.5B-Instruct-q4f16_1-MLC';
})[0];
assert(qwen.model_lib.indexOf('/Qwen2-0.5B-Instruct-q4f16_1_cs1k-webgpu.wasm') !== -1, 'Qwen 0.5B keeps the vendored Qwen2 wasm file name');

assert(chat.indexOf('appConfig: webllmAppConfig') !== -1, 'MLCEngine receives the pinned appConfig');
assert(chat.indexOf('hasModelInCache(opt.id, webllmAppConfig)') !== -1, 'cache checks use the pinned appConfig');
assert(chat.indexOf('deleteModelAllInfoInCache(opt.id, webllmAppConfig)') !== -1, 'delete uses the pinned appConfig');
assert(chat.indexOf("const WEBLLM_DEFAULT_BASE = 'Llama-3.2-1B-Instruct'") !== -1, 'default base stays Llama-3.2-1B-Instruct');
assert(chat.indexOf("let llmModelId = 'Llama-3.2-1B-Instruct-q4f16_1-MLC'") !== -1, 'initial id stays Llama-3.2-1B q4f16');

console.log('PIN-MODELS-1 checks passed');
