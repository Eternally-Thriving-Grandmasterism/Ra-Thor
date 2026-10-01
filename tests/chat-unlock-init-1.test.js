/* UNLOCK-INIT-1: the post-load tail runs once, after a plain load or a good unlock. */
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

function body(startMark, endMark) {
  var start = chat.indexOf(startMark);
  var end = chat.indexOf(endMark, start + startMark.length);
  assert(start !== -1 && end > start, 'missing ' + startMark);
  return chat.slice(start, end);
}

assert(chat.indexOf('<think>') === -1, 'chat.js must not gain a think tag');
assert(chat.indexOf('window.confirm') === -1, 'chat.js must not call window.confirm');
assert(chat.indexOf("./vendor/web-llm/0.2.85/index.js") !== -1, 'WebLLM pin stays 0.2.85');

var unlockFn = body('async function tryUnlock', 'function warnIfPlaintextUnderFlag');
var speechFn = body('function initSpeechRecognition', 'function toggleMic');
var initMark = '// ─── Init ';
var initAt = chat.indexOf(initMark);
var initEnd = chat.lastIndexOf('})();');
assert(initAt !== -1 && initEnd > initAt, 'init block must be marked');
var init = chat.slice(initAt, initEnd);

assert(unlockFn.indexOf('loadStore()') === -1, 'unlock must not run a plain loadStore');
assert(unlockFn.indexOf('warnIfPlaintextUnderFlag') === -1, 'unlock must not run the plaintext notice');
var renderAt = unlockFn.indexOf('renderHistory();');
var unlockCall = unlockFn.indexOf('postLoadInit();');
assert(renderAt !== -1 && unlockCall > renderAt, 'a good unlock calls postLoadInit after renderHistory');
assert(unlockFn.indexOf('setLockedAppInert(false)') < unlockCall, 'inert is lifted before postLoadInit');
var failBranch = unlockFn.slice(unlockFn.indexOf('} else {'));
assert(failBranch.indexOf('postLoadInit') === -1, 'a failed unlock never calls postLoadInit');

var tail = body('async function postLoadInit', "Optional Passphrase Encryption ready");
var guardAt = tail.indexOf('postLoadInitStarted = true;');
var firstAwait = tail.indexOf('await ');
assert(guardAt !== -1 && firstAwait > guardAt, 'guard is set before any await');
assert(tail.indexOf('localStorage') === -1 && tail.indexOf('indexedDB') === -1 && tail.indexOf('saveStore') === -1, 'postLoadInit writes nothing');
var order = ['initSpeechRecognition();', 'await detectLocalLlmSupport();', 'await initWebllmPicker();', 'applyChatSurfaceDir();', 'setBackendUI(false);', 'onvoiceschanged', 'console.log('];
var last = -1;
order.forEach(function (mark) {
  var at = tail.indexOf(mark);
  assert(at > last, 'postLoadInit keeps the tail order at ' + mark);
  last = at;
});

var plain = init.slice(init.indexOf('await loadStore()'));
var plainOrder = ['await loadStore()', 'refreshSessionSelect();', 'renderHistory();', 'warnIfPlaintextUnderFlag();', 'await postLoadInit();'];
last = -1;
plainOrder.forEach(function (mark) {
  var at = plain.indexOf(mark);
  assert(at > last, 'plain load order at ' + mark);
  last = at;
});

function tick() {
  return new Promise(function (resolve) { setImmediate(resolve); });
}

async function flush() {
  for (var i = 0; i < 10; i++) await tick();
}

function makeSandbox(encrypted) {
  var counts = { recognition: 0, picker: 0, detect: 0, voicesHook: 0, surface: 0, backendFalse: 0, log: 0, plainLoad: 0, warn: 0, refresh: 0, render: 0 };
  var listeners = {};
  var voicesHandler = null;
  var classes = function () {
    var set = {};
    return {
      add: function (c) { set[c] = true; },
      remove: function (c) { delete set[c]; },
      contains: function (c) { return !!set[c]; },
      toggle: function () {}
    };
  };
  var speech = { getVoices: function () { return []; } };
  Object.defineProperty(speech, 'onvoiceschanged', {
    get: function () { return voicesHandler; },
    set: function (fn) { counts.voicesHook++; voicesHandler = fn; }
  });
  function FakeRecognition() {
    counts.recognition++;
  }
  var win = {
    SpeechRecognition: FakeRecognition,
    speechSynthesis: speech,
    addEventListener: function (type, fn) {
      (listeners[type] = listeners[type] || []).push(fn);
    }
  };
  var sandbox = {
    window: win,
    document: { querySelector: function () { return null; } },
    console: { log: function () { counts.log++; }, warn: function () {}, error: function () {} },
    localStorage: { getItem: function () { return null; }, setItem: function () {} },
    setTimeout: setTimeout,
    counts: counts,
    listeners: listeners
  };
  vm.createContext(sandbox);
  var harness = [
    'var unlockOverlay = { classList: (' + classes.toString() + ')() };',
    'var unlockError = { classList: (' + classes.toString() + ')() };',
    'var unlockPassphrase = { value: "" };',
    'var micBtn = null, chatInput = null, localLlmBtn = null, backendSettings = null, webllmPicker = null;',
    'var recognition = null, isListening = false;',
    'var isEncrypted = false, cryptoKey = null, cryptoSalt = null;',
    'var llmSupported = false, webllmPhonePath = false, llmProbed = false, llmReady = false, llmLoading = false;',
    'var ENCRYPT_FLAG = "flag";',
    'function flagAfterUnlock(f) { return f; }',
    'function isStoreEncrypted() { return ' + (encrypted ? 'true' : 'false') + '; }',
    'async function loadStore(pass) { if (pass === undefined) { counts.plainLoad++; return true; } return pass === "right"; }',
    'function refreshSessionSelect() { counts.refresh++; }',
    'function renderHistory() { counts.render++; }',
    'function warnIfPlaintextUnderFlag() { counts.warn++; }',
    'function addNotice() {}',
    'function loadSettings() {}',
    'function renderLocalServerPresets() {}',
    'function renderNetMode() {}',
    'function sendMessage() {}',
    'async function detectLocalLlmSupport() { counts.detect++; await null; return { supported: true, phone: false }; }',
    'async function initWebllmPicker() { counts.picker++; await null; return true; }',
    'function updateLlmUI() {}',
    'function applyChatSurfaceDir() { counts.surface++; }',
    'function setBackendUI(on) { if (on === false) counts.backendFalse++; }',
    'function setLockedAppInert() {}'
  ].join('\n');
  vm.runInContext(harness + '\n' + unlockFn + '\n' + speechFn + '\n' + init +
    '\nthis.api = { tryUnlock: tryUnlock, setPass: function (p) { unlockPassphrase.value = p; }, guard: function () { return typeof postLoadInitStarted === "undefined" ? "missing" : postLoadInitStarted; }, recognition: function () { return recognition; } };', sandbox);
  return sandbox;
}

function assertOnce(counts, label) {
  assert(counts.recognition === 1, label + ': speech recognition built once, got ' + counts.recognition);
  assert(counts.detect === 1, label + ': local LLM probe runs once, got ' + counts.detect);
  assert(counts.picker === 1, label + ': picker init runs once, got ' + counts.picker);
  assert(counts.voicesHook === 1, label + ': voices hook set once, got ' + counts.voicesHook);
  assert(counts.surface === 1, label + ': surface dir applied once, got ' + counts.surface);
  assert(counts.backendFalse === 1, label + ': setBackendUI(false) once, got ' + counts.backendFalse);
  assert(counts.log === 1, label + ': ready log once, got ' + counts.log);
}

async function main() {
  var locked = makeSandbox(true);
  var loadHandlers = locked.listeners.DOMContentLoaded || [];
  assert(loadHandlers.length === 1, 'one DOMContentLoaded handler');
  await loadHandlers[0]();
  await flush();
  assert(locked.counts.recognition === 0 && locked.counts.picker === 0, 'locked load waits for unlock');
  assert(locked.counts.plainLoad === 0, 'locked load does not read the plain store');
  assert(locked.api.guard() === false, 'guard is clear while locked: ' + locked.api.guard());

  locked.api.setPass('wrong');
  await locked.api.tryUnlock();
  await flush();
  assert(locked.api.guard() === false, 'a failed unlock does not set the guard');
  assert(locked.counts.recognition === 0 && locked.counts.picker === 0 && locked.counts.voicesHook === 0, 'a failed unlock runs nothing');

  locked.api.setPass('right');
  await locked.api.tryUnlock();
  assert(locked.api.guard() === true, 'guard is set as soon as postLoadInit starts');
  await flush();
  assertOnce(locked.counts, 'after unlock');
  var rec = locked.api.recognition();
  assert(rec && typeof rec.onstart === 'function' && typeof rec.onresult === 'function' && typeof rec.onend === 'function' && typeof rec.onerror === 'function', 'speech handlers are wired after unlock');
  assert(locked.counts.plainLoad === 0 && locked.counts.warn === 0, 'unlock does not plain-load or warn');
  assert(locked.counts.refresh === 1 && locked.counts.render === 1, 'unlock keeps its own refresh and render');

  await locked.api.tryUnlock();
  await flush();
  assertOnce(locked.counts, 'after second unlock');
  assert(locked.api.recognition() === rec, 'second unlock keeps the same recognition handlers');
  assert(locked.counts.refresh === 2 && locked.counts.render === 2, 'second unlock still refreshes the list');

  var plainBox = makeSandbox(false);
  await plainBox.listeners.DOMContentLoaded[0]();
  await flush();
  assertOnce(plainBox.counts, 'plain load');
  assert(plainBox.counts.plainLoad === 1 && plainBox.counts.warn === 1, 'plain load reads the store and checks the flag once');
  assert(plainBox.api.guard() === true, 'plain load sets the guard');
}

main().then(function () {
  console.log('chat-unlock-init-1 ok');
}).catch(function (err) {
  console.error(err);
  process.exit(1);
});
