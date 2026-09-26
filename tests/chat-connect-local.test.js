/* CONNECT-LOCAL: Local Server presets fill endpoint + model and do not connect. */
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
var en = read('i18n/en.js');
var sw = read('sw.js');

var promptMarker = 'const SYSTEM_PROMPT = `';
var promptAt = chat.indexOf(promptMarker);
var promptEnd = chat.indexOf('`;', promptAt);
var prompt = chat.slice(promptAt + promptMarker.length, promptEnd);
var promptHash = crypto.createHash('sha256').update(prompt).digest('hex');
assert(promptHash === '1fc78b6d442e43494de3bb52adc5307eb3c7861476cd0d838af0c3fcd61def61', 'SYSTEM_PROMPT hash changed: ' + promptHash);
assert(chat.indexOf("./vendor/web-llm/0.2.85/index.js") !== -1, 'WebLLM pin stays 0.2.85');
assert(chat.indexOf('const PHONE_MAX_VRAM_MB = 1200;') !== -1, 'phone adapter cap stays 1200');
assert(sw.indexOf("var LOCK = '20260924a';") !== -1, 'service worker LOCK stays 20260924a');
assert(chat.indexOf("const BACKEND_KEY = 'rathor-local-backend-v1'") !== -1, 'backend storage key stays');

var pure = chat.slice(chat.indexOf('/* chat-models-1-pure */'), chat.indexOf('/* chat-models-1-pure-end */'));
assert(pure.indexOf('function localServerPresets') !== -1, 'presets live in the pure block');
assert(pure.indexOf('function fillFromLocalServerPreset') !== -1, 'fill helper lives in the pure block');
var sandbox = { URL: URL };
vm.createContext(sandbox);
vm.runInContext(pure + '\nthis.api = { localServerPresets: localServerPresets, fillFromLocalServerPreset: fillFromLocalServerPreset, localServerEndpointAllowed: localServerEndpointAllowed };', sandbox);
var api = sandbox.api;

var expect = [
  { id: 'ollama', label: 'Ollama', endpoint: 'http://localhost:11434/v1', model: 'llama3.2' },
  { id: 'lmstudio', label: 'LM Studio', endpoint: 'http://localhost:1234/v1', model: 'local-model' },
  { id: 'llamacpp', label: 'llama.cpp', endpoint: 'http://localhost:8080/v1', model: 'llama' },
  { id: 'vllm', label: 'vLLM', endpoint: 'http://localhost:8000/v1', model: 'default' }
];
var presets = api.localServerPresets();
assert(presets.length === 4, 'four local presets');
expect.forEach(function (row, i) {
  assert(presets[i].id === row.id, 'preset order ' + row.id);
  assert(presets[i].endpoint === row.endpoint, row.id + ' endpoint');
  assert(presets[i].model === row.model, row.id + ' model');
  assert(presets[i].label === row.label, row.id + ' label');
  assert(!Object.prototype.hasOwnProperty.call(presets[i], 'apiKey'), row.id + ' stores no api key');
  assert(api.localServerEndpointAllowed('offline-only', presets[i].endpoint) === true, row.id + ' stays loopback');
  var filled = api.fillFromLocalServerPreset(row.id, { endpoint: 'http://10.0.0.8/v1', model: 'untouched', connected: false });
  assert(filled.applied === true, row.id + ' fills');
  assert(filled.endpoint === row.endpoint, row.id + ' fill endpoint');
  assert(filled.model === row.model, row.id + ' fill model');
  assert(filled.connected === false, row.id + ' must not connect');
  assert(filled.effect === null, row.id + ' must not start a connect effect');
  console.log(row.id + ' ' + row.endpoint + ' model=' + row.model);
});

var stayed = api.fillFromLocalServerPreset('ollama', { endpoint: 'http://127.0.0.1:9/v1', model: 'old', connected: true });
assert(stayed.connected === true, 'picking a preset must not drop an existing connection');
assert(stayed.effect === null, 'picking a preset must not reconnect');
assert(stayed.endpoint === 'http://localhost:11434/v1', 'ollama endpoint still fills while connected');

var unknown = api.fillFromLocalServerPreset('remote-cloud', { endpoint: 'http://localhost:11434/v1', model: 'llama3.2', connected: false });
assert(unknown.applied === false && unknown.connected === false && unknown.effect === null, 'unknown preset does not connect');
assert(unknown.endpoint === 'http://localhost:11434/v1' && unknown.model === 'llama3.2', 'unknown preset leaves the fields');

assert(api.localServerEndpointAllowed('offline-only', 'https://api.example.com/v1') === false, 'offline only still blocks a non-loopback endpoint');
assert(api.localServerEndpointAllowed('network-on', 'https://api.example.com/v1') === true, 'network on still allows a non-loopback endpoint');

var panelStart = html.indexOf('id="backend-settings"');
var panelEnd = html.indexOf('id="local-llm-progress"');
assert(panelStart !== -1 && panelEnd > panelStart, 'local server panel exists');
var panel = html.slice(panelStart, panelEnd);
expect.forEach(function (row) {
  assert(panel.indexOf('data-preset="' + row.id + '"') !== -1, 'panel button ' + row.id);
  assert(panel.indexOf('>' + row.label + '<') !== -1, 'panel label ' + row.label);
});
assert(panel.indexOf('id="local-server-presets"') !== -1, 'preset group exists');
assert(panel.indexOf('id="backend-connect-btn"') !== -1, 'Connect stays a separate control');
assert(panel.indexOf('id="backend-disconnect-btn"') !== -1, 'Disconnect stays in the panel');
assert(panel.indexOf('id="backend-status"') !== -1, 'status line stays');
assert(panel.indexOf('Not connected') !== -1, 'status still starts at Not connected');
assert(panel.toLowerCase().indexOf('api-key') === -1, 'no api key field in the local panel');
assert(panel.toLowerCase().indexOf('api_key') === -1, 'no api_key field in the local panel');
assert(panel.indexOf('type="password"') === -1, 'no password field in the local panel');

var selectStart = chat.indexOf('function selectLocalServerPreset');
var selectEnd = chat.indexOf('function setBackendUI');
assert(selectStart !== -1 && selectEnd > selectStart, 'selectLocalServerPreset exists');
var selectFn = chat.slice(selectStart, selectEnd);
assert(selectFn.indexOf('fillFromLocalServerPreset') !== -1, 'preset select fills from the preset table');
assert(selectFn.indexOf('connectBackend') === -1, 'preset select must not call connect');
assert(selectFn.indexOf('saveBackendConfig') === -1, 'preset select must not persist a connection');
assert(selectFn.indexOf('fetch(') === -1, 'preset select must not reach the network');

var clickStart = chat.indexOf('localServerPresetsEl.addEventListener');
var clickEnd = chat.indexOf("if (backendConnectBtn) backendConnectBtn.addEventListener('click', connectBackend)");
assert(clickStart !== -1 && clickEnd > clickStart, 'preset click is wired before Connect');
var clickFn = chat.slice(clickStart, clickEnd);
assert(clickFn.indexOf('selectLocalServerPreset') !== -1, 'preset click fills fields');
assert(clickFn.indexOf('connectBackend') === -1, 'preset click must not connect');
assert(chat.indexOf("backendConnectBtn.addEventListener('click', connectBackend)") !== -1, 'Connect remains the confirm');
assert(chat.indexOf("backendDisconnectBtn.addEventListener('click', disconnectBackend)") !== -1, 'Disconnect remains wired');

assert(en.indexOf('"chatPresetOllama": "Ollama"') !== -1, 'English Ollama label');
assert(en.indexOf('"chatPresetLmStudio": "LM Studio"') !== -1, 'English LM Studio label');
assert(en.indexOf('"chatPresetLlamaCpp": "llama.cpp"') !== -1, 'English llama.cpp label');
assert(en.indexOf('"chatPresetVllm": "vLLM"') !== -1, 'English vLLM label');
assert(en.indexOf('"chatPresetNote": "Pick a preset, then Connect. You can edit both fields."') !== -1, 'English preset note');
assert(en.indexOf('"chatPresetGroup": "Local server presets"') !== -1, 'English preset group label');
assert(chat.indexOf("chatLabel('chatPresetNote'") !== -1, 'preset note is read from the English pack');

console.log('chat-connect-local: ok');
