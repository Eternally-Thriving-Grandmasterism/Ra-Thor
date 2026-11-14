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

var pure = chat.slice(chat.indexOf('/* chat-models-1-pure */'), chat.indexOf('/* chat-models-1-pure-end */'));
var cacheSandbox = { URL: URL };
vm.createContext(cacheSandbox);
vm.runInContext(pure + '\nthis.cacheApi = { requestMatchesModel: requestMatchesModel, requestMatchesPurge: requestMatchesPurge, legacyResolveMainPrefix: legacyResolveMainPrefix, legacyWasmLibUrl: legacyWasmLibUrl, purgeOwnedModelEntries: purgeOwnedModelEntries, modelFilesRemain: modelFilesRemain, phaseAfterLocalDelete: phaseAfterLocalDelete, retiredGemmaTargets: retiredGemmaTargets, storesAfterGemmaFallback: storesAfterGemmaFallback };', cacheSandbox);
var cacheApi = cacheSandbox.cacheApi;

var llamaId = 'Llama-3.2-1B-Instruct-q4f16_1-MLC';
var qwenId = 'Qwen2.5-0.5B-Instruct-q4f16_1-MLC';
var llamaSha = api.WEBLLM_HF_REVISIONS[llamaId];
var qwenSha = api.WEBLLM_HF_REVISIONS[qwenId];
var wasmSha = api.WEBLLM_WASM_REVISION;
function hf(id, rev, file) {
  return 'https://huggingface.co/mlc-ai/' + id + '/resolve/' + rev + '/' + file;
}
function wasm(rev, file) {
  return 'https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/' + rev + '/web-llm-models/v0_2_84/base/' + file;
}
var pinnedModel = hf(llamaId, llamaSha, '');
var pinnedWasm = wasm(wasmSha, 'Llama-3.2-1B-Instruct-q4f16_1_cs1k-webgpu.wasm');
var legacyWasm = wasm('main', 'Llama-3.2-1B-Instruct-q4f16_1_cs1k-webgpu.wasm');
var qwenWasm = wasm('main', 'Qwen2-0.5B-Instruct-q4f16_1_cs1k-webgpu.wasm');
var pinnedTensor = hf(llamaId, llamaSha, 'tensor-cache.json');
var legacyTensor = hf(llamaId, 'main', 'tensor-cache.json');
var pinnedShard = hf(llamaId, llamaSha, 'params_shard_0.bin');
var legacyShard = hf(llamaId, 'main', 'params_shard_0.bin');
var pinnedConfig = hf(llamaId, llamaSha, 'mlc-chat-config.json');
var legacyConfig = hf(llamaId, 'main', 'mlc-chat-config.json');
var qwenLegacy = hf(qwenId, 'main', 'tensor-cache.json');
var qwenPinned = hf(qwenId, qwenSha, 'tensor-cache.json');
var siblingLegacy = hf(llamaId + '-1k', 'main', 'tensor-cache.json');

assert(cacheApi.legacyResolveMainPrefix(pinnedModel) === hf(llamaId, 'main', ''), 'legacy weight prefix is the unpinned revision of the same id');
assert(cacheApi.legacyWasmLibUrl(pinnedWasm) === legacyWasm, 'legacy wasm keeps the path and uses the unpinned git ref');
assert(cacheApi.requestMatchesModel(legacyTensor, pinnedModel, pinnedWasm) === false, 'a legacy weight is not cached for the pinned model');
assert(cacheApi.requestMatchesModel(legacyWasm, pinnedModel, pinnedWasm) === false, 'a legacy wasm is not cached for the pinned model');
assert(cacheApi.requestMatchesModel(pinnedTensor, pinnedModel, pinnedWasm) === true, 'a pinned weight still counts as cached');
assert(cacheApi.requestMatchesModel(pinnedWasm, pinnedModel, pinnedWasm) === true, 'a pinned wasm still counts as cached');
assert(cacheApi.modelFilesRemain({
  'webllm/model': [legacyTensor, legacyShard],
  'webllm/wasm': [legacyWasm],
  'webllm/config': [legacyConfig]
}, pinnedModel, pinnedWasm) === false, 'legacy entries alone do not mark the pinned model cached');
assert(cacheApi.requestMatchesPurge(legacyTensor, pinnedModel, pinnedWasm) === true, 'delete matches a legacy weight');
assert(cacheApi.requestMatchesPurge(legacyWasm, pinnedModel, pinnedWasm) === true, 'delete matches a legacy wasm');
assert(cacheApi.requestMatchesPurge(qwenLegacy, pinnedModel, pinnedWasm) === false, 'delete does not match another model');
assert(cacheApi.requestMatchesPurge(siblingLegacy, pinnedModel, pinnedWasm) === false, 'delete does not match a prefix sibling');

var cacheStores = {
  'webllm/model': [pinnedTensor, pinnedShard, legacyTensor, legacyShard, qwenLegacy, qwenPinned],
  'webllm/wasm': [pinnedWasm, legacyWasm, qwenWasm],
  'webllm/config': [pinnedConfig, legacyConfig],
  'webllm/script': [legacyTensor],
  'rathor-core-20260924a': [legacyShard]
};
var idbStores = {
  'webllm/model': [pinnedShard, legacyShard, qwenLegacy],
  'webllm/wasm': [pinnedWasm, legacyWasm, qwenWasm],
  'webllm/config': [pinnedConfig, legacyConfig],
  'notes-db': [legacyTensor]
};
var afterCache = cacheApi.purgeOwnedModelEntries(cacheStores, pinnedModel, pinnedWasm);
var afterIdb = cacheApi.purgeOwnedModelEntries(idbStores, pinnedModel, pinnedWasm);
function gone(list, url) { return list.indexOf(url) === -1; }
function kept(list, url) { return list.indexOf(url) !== -1; }
assert(gone(afterCache['webllm/model'], pinnedTensor) && gone(afterCache['webllm/model'], legacyTensor), 'Cache API delete removes pinned and legacy weights');
assert(gone(afterCache['webllm/model'], pinnedShard) && gone(afterCache['webllm/model'], legacyShard), 'Cache API delete removes pinned and legacy shards');
assert(gone(afterCache['webllm/wasm'], pinnedWasm) && gone(afterCache['webllm/wasm'], legacyWasm), 'Cache API delete removes pinned and legacy wasm');
assert(gone(afterCache['webllm/config'], pinnedConfig) && gone(afterCache['webllm/config'], legacyConfig), 'Cache API delete removes pinned and legacy config');
assert(kept(afterCache['webllm/model'], qwenLegacy) && kept(afterCache['webllm/model'], qwenPinned), 'Cache API delete leaves the other model');
assert(kept(afterCache['webllm/wasm'], qwenWasm), 'Cache API delete leaves the other wasm');
assert(afterCache['webllm/script'][0] === legacyTensor, 'the script cache is not a model cache');
assert(afterCache['rathor-core-20260924a'][0] === legacyShard, 'a cache this page does not own stays untouched');
assert(gone(afterIdb['webllm/model'], pinnedShard) && gone(afterIdb['webllm/model'], legacyShard), 'IndexedDB delete removes pinned and legacy weights');
assert(gone(afterIdb['webllm/wasm'], pinnedWasm) && gone(afterIdb['webllm/wasm'], legacyWasm), 'IndexedDB delete removes pinned and legacy wasm');
assert(gone(afterIdb['webllm/config'], pinnedConfig) && gone(afterIdb['webllm/config'], legacyConfig), 'IndexedDB delete removes pinned and legacy config');
assert(kept(afterIdb['webllm/model'], qwenLegacy) && kept(afterIdb['webllm/wasm'], qwenWasm), 'IndexedDB delete leaves the other model');
assert(afterIdb['notes-db'][0] === legacyTensor, 'an unrelated IndexedDB database stays untouched');
assert(cacheApi.modelFilesRemain(afterCache, pinnedModel, pinnedWasm) === false, 'after delete the pinned model is not cached');
assert(cacheApi.phaseAfterLocalDelete(cacheStores, pinnedModel, pinnedWasm) === 'absent', 'delete leaves the row absent');
assert(cacheStores['webllm/model'].indexOf(legacyTensor) !== -1, 'the fixture itself is not mutated');

var gemma16 = hf('gemma-2-2b-it-q4f16_1-MLC', 'main', 'params_shard_0.bin');
var gemma32 = hf('gemma-2-2b-it-q4f32_1-MLC', 'main', 'tensor-cache.json');
var gemma16wasm = wasm('main', 'gemma-2-2b-it-q4f16_1_cs1k-webgpu.wasm');
var gemma32wasm = wasm('main', 'gemma-2-2b-it-q4f32_1_cs1k-webgpu.wasm');
var gemmaConfig = hf('gemma-2-2b-it-q4f16_1-MLC', 'main', 'mlc-chat-config.json');
var targets = cacheApi.retiredGemmaTargets();
assert(targets.length === 2, 'gemma fallback covers both weights');
assert(targets[0].model === hf('gemma-2-2b-it-q4f16_1-MLC', 'main', ''), 'gemma q4f16 target is the unpinned repo');
assert(targets[1].model_lib === gemma32wasm, 'gemma q4f32 target is the vendored wasm file');
var gemmaCaches = {
  'webllm/model': [gemma16, gemma32, legacyTensor],
  'webllm/wasm': [gemma16wasm, gemma32wasm, legacyWasm],
  'webllm/config': [gemmaConfig],
  'webllm/script': [gemma16],
  'rathor-core-20260924a': [gemma32]
};
var gemmaIdb = {
  'webllm/model': [gemma16, gemma32, qwenLegacy],
  'webllm/wasm': [gemma16wasm, gemma32wasm, qwenWasm],
  'other-site': [gemma16]
};
var gemmaCleared = cacheApi.storesAfterGemmaFallback(gemmaCaches, gemmaIdb, 'gemma-2-2b-it');
assert(gone(gemmaCleared.caches['webllm/model'], gemma16) && gone(gemmaCleared.caches['webllm/model'], gemma32), 'gemma fallback purges both legacy weight caches');
assert(kept(gemmaCleared.caches['webllm/model'], legacyTensor), 'gemma fallback leaves another model weight');
assert(gone(gemmaCleared.caches['webllm/wasm'], gemma16wasm) && gone(gemmaCleared.caches['webllm/wasm'], gemma32wasm), 'gemma fallback purges both legacy wasm files');
assert(kept(gemmaCleared.caches['webllm/wasm'], legacyWasm), 'gemma fallback leaves another wasm');
assert(gone(gemmaCleared.caches['webllm/config'], gemmaConfig), 'gemma fallback purges the legacy config');
assert(gemmaCleared.caches['webllm/script'][0] === gemma16, 'gemma fallback does not touch the script cache');
assert(gemmaCleared.caches['rathor-core-20260924a'][0] === gemma32, 'gemma fallback does not touch another site cache');
assert(gone(gemmaCleared.indexedDB['webllm/model'], gemma16) && gone(gemmaCleared.indexedDB['webllm/model'], gemma32), 'gemma fallback purges IndexedDB weights');
assert(kept(gemmaCleared.indexedDB['webllm/model'], qwenLegacy), 'gemma fallback leaves another IndexedDB model');
assert(gone(gemmaCleared.indexedDB['webllm/wasm'], gemma16wasm) && gone(gemmaCleared.indexedDB['webllm/wasm'], gemma32wasm), 'gemma fallback purges IndexedDB wasm');
assert(gemmaCleared.indexedDB['other-site'][0] === gemma16, 'gemma fallback does not touch an unrelated database');
var gemmaKept = cacheApi.storesAfterGemmaFallback(gemmaCaches, gemmaIdb, 'Llama-3.2-1B-Instruct');
assert(gemmaKept.caches === gemmaCaches && gemmaKept.indexedDB === gemmaIdb, 'a non-gemma saved choice does not purge');

var cachedFn = chat.slice(chat.indexOf('async function modelFilesCached'), chat.indexOf('async function purgeCacheApiModel'));
assert(cachedFn.indexOf('requestMatchesModel') !== -1 && cachedFn.indexOf('requestMatchesPurge') === -1, 'cache detection stays on the pinned record');
var purgeFn = chat.slice(chat.indexOf('async function purgeModelLeftovers'), chat.indexOf('async function purgeRetiredGemmaCaches'));
assert(purgeFn.indexOf('purgeCacheApiModel') !== -1 && purgeFn.indexOf('purgeIndexedDbModel') !== -1, 'delete sweeps Cache API and IndexedDB');
var idbFn = chat.slice(chat.indexOf('async function purgeIndexedDbModel'), chat.indexOf('async function purgeModelLeftovers'));
assert(idbFn.indexOf('indexedDB.databases') !== -1 && idbFn.indexOf("transaction(['urls']") !== -1, 'IndexedDB sweep lists existing databases and the urls store');
assert(idbFn.indexOf('isWebllmOwnedCache') !== -1 && idbFn.indexOf('requestMatchesPurge') !== -1, 'IndexedDB sweep stays on owned model urls');
var initFn = chat.slice(chat.indexOf('async function initWebllmPicker'), chat.indexOf('async function generateWithLocalLLM'));
assert(initFn.indexOf("rawStored === 'gemma-2-2b-it'") !== -1 && initFn.indexOf('purgeRetiredGemmaCaches') !== -1, 'a saved gemma choice purges the legacy gemma cache');
assert(initFn.indexOf('addNotice') === -1, 'the gemma fallback adds no on-screen text');

console.log('PIN-MODELS-1 checks passed');
