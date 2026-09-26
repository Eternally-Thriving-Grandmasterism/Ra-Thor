/* REPLY-BUDGET-1: context budget, one user turn, notices stay out, length marks the turn. */
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
var en = read('i18n/en.js');

assert(chat.indexOf('<think>') === -1, 'chat.js must not gain a think tag');
assert(chat.indexOf('window.confirm') === -1, 'chat.js must not call window.confirm');
assert(chat.indexOf('max_tokens: 500') === -1, 'old WebLLM max_tokens 500 must be gone');
assert(chat.indexOf('max_tokens: 900') === -1, 'old Local Server max_tokens 900 must be gone');
assert(chat.indexOf('content: userText') === -1, 'user text must not be pushed a second time');
assert(chat.indexOf('loadedModelIdToChatConfig') !== -1, 'context is read from the engine chat config');
assert(chat.indexOf('Never invent a 4096') !== -1, 'context size must not be assumed');

var pureStart = chat.indexOf('/* chat-reply-budget-pure */');
var pureEnd = chat.indexOf('/* chat-reply-budget-pure-end */');
assert(pureStart !== -1 && pureEnd > pureStart, 'reply budget pure block must be marked');
var pure = chat.slice(pureStart, pureEnd);
assert(pure.indexOf('function replyTokenBudget') !== -1, 'budget math must be a function');
assert(pure.indexOf('function modelMessagesFromHistory') !== -1, 'message builder must be a function');
assert(pure.indexOf('function turnWithFinish') !== -1, 'finish_reason mark must be a function');

var sandbox = {};
vm.createContext(sandbox);
vm.runInContext(pure + '\nthis.api = { replyTokenBudget: replyTokenBudget, contextTokensFromSources: contextTokensFromSources, promptCharsOfMessages: promptCharsOfMessages, modelMessagesFromHistory: modelMessagesFromHistory, planReply: planReply, choiceFinishReason: choiceFinishReason, turnWithFinish: turnWithFinish };', sandbox);
var api = sandbox.api;

var floor = api.replyTokenBudget({ contextTokens: 320, promptChars: 0, messageCount: 0, ceiling: 2048 });
assert(floor.send === true && floor.maxTokens === 256, 'floor is 256: ' + JSON.stringify(floor));

var under = api.replyTokenBudget({ contextTokens: 319, promptChars: 0, messageCount: 0, ceiling: 2048 });
assert(under.send === false && under.maxTokens === 0, 'under 256 does not send: ' + JSON.stringify(under));

var cap = api.replyTokenBudget({ contextTokens: 100000, promptChars: 0, messageCount: 0, ceiling: 2048 });
assert(cap.send === true && cap.maxTokens === 2048, 'ceiling is 2048: ' + JSON.stringify(cap));

var mid = api.replyTokenBudget({ contextTokens: 1000, promptChars: 30, messageCount: 4, ceiling: 2048 });
assert(mid.send === true && mid.maxTokens === 894, 'mid budget is room, not the ceiling: ' + JSON.stringify(mid));

var packed = api.replyTokenBudget({ contextTokens: 400, promptChars: 300, messageCount: 10 });
assert(packed.send === false && packed.maxTokens === 0, 'context full when room is under 256: ' + JSON.stringify(packed));

var assumed = api.replyTokenBudget({ contextTokens: null, promptChars: 100000, messageCount: 40, ceiling: 2048 });
assert(assumed.send === true && assumed.maxTokens === 2048 && assumed.contextTokens === null, 'unknown context uses the 2048 ceiling: ' + JSON.stringify(assumed));

var def = api.replyTokenBudget({ contextTokens: 9000, promptChars: 0, messageCount: 0 });
assert(def.maxTokens === 2048, 'default ceiling is 2048: ' + def.maxTokens);

assert(api.contextTokensFromSources(8192, 3000, { overrides: { context_window_size: 4096 } }) === 8192, 'engine config wins');
assert(api.contextTokensFromSources({ context_window_size: 5120 }, 3000, { overrides: { context_window_size: 4096 } }) === 5120, 'chat config object is read');
assert(api.contextTokensFromSources(null, 3000, { overrides: { context_window_size: 4096 } }) === 3000, 'ChatOptions at load are next');
assert(api.contextTokensFromSources(-1, 0, { overrides: { context_window_size: 4096 } }) === 4096, 'prebuilt record is the fallback');
assert(api.contextTokensFromSources(null, null, { context_window_size: 2048 }) === 2048, 'top-level record value is the fallback');
assert(api.contextTokensFromSources(null, null, null) === null, 'missing sources stay unknown');

var history = [
  { role: 'rathor', text: 'Document “notes.txt” injected into context.', notice: true },
  { role: 'user', text: 'first' },
  { role: 'rathor', text: 'Context is full. Start a new chat or remove documents to continue.', notice: true },
  { role: 'user', text: 'second question' },
  { role: 'rathor', text: 'an answer', cutOff: true }
];
var built = api.modelMessagesFromHistory('SYS', history, 14);
assert(built[0].role === 'system' && built[0].content === 'SYS', 'system prompt is first');
var userTurns = built.filter(function (m) { return m.role === 'user' && m.content === 'second question'; });
assert(userTurns.length === 1, 'the latest user turn is sent once: ' + JSON.stringify(built));
assert(built.filter(function (m) { return m.role === 'user'; }).length === 2, 'each saved user turn is sent once');
assert(built.some(function (m) { return String(m.content).indexOf('injected into context') !== -1; }) === false, 'doc notice stays out');
assert(built.some(function (m) { return String(m.content).indexOf('Context is full') !== -1; }) === false, 'context-full notice stays out');
assert(built.some(function (m) { return m.content === 'an answer'; }) === true, 'assistant reply stays in');
assert(built.some(function (m) { return String(m.content).indexOf('Reply stopped') !== -1; }) === false, 'cut-off line is not a model message');

var plan = api.planReply('SYS', history, 14, 4096);
assert(plan.messages.length === built.length, 'plan sends the built messages');
assert(plan.messageCount === plan.messages.length, 'messageCount is the number sent');
assert(plan.promptChars === api.promptCharsOfMessages(plan.messages), 'promptChars counts the sent messages');
assert(plan.promptChars === ('SYS'.length + 'first'.length + 'second question'.length + 'an answer'.length), 'promptChars counts system, history, and the user text once');
assert(plan.budget.send === true && plan.budget.maxTokens <= 2048 && plan.budget.maxTokens >= 256, 'plan budget is inside the clamp');

var longHist = [];
for (var i = 0; i < 20; i++) longHist.push({ role: 'user', text: 'u' + i });
longHist.push({ role: 'rathor', text: 'Local Server connected.', notice: true });
var limited = api.modelMessagesFromHistory('S', longHist, 10);
assert(limited.length === 11, 'limit keeps ten conversational turns plus system');
assert(limited[limited.length - 1].content === 'u19', 'the newest user turn survives the slice');
assert(limited.filter(function (m) { return m.content === 'u19'; }).length === 1, 'slice does not duplicate the user turn');
assert(limited.some(function (m) { return String(m.content).indexOf('Local Server connected') !== -1; }) === false, 'connect notice stays out of the slice');

var cut = api.turnWithFinish('partial reply', api.choiceFinishReason({ finish_reason: 'length' }));
assert(cut.cutOff === true && cut.text === 'partial reply' && cut.role === 'rathor', 'length marks the assistant turn: ' + JSON.stringify(cut));
var stopped = api.turnWithFinish('done', 'stop');
assert(stopped.cutOff !== true, 'stop does not mark a cut-off');
assert(api.choiceFinishReason({ finish_reason: null }) === null, 'empty finish_reason is ignored');
assert(api.turnWithFinish('x', null).cutOff !== true, 'missing finish_reason does not mark a cut-off');

function body(startMark, endMark) {
  var start = chat.indexOf(startMark);
  var end = chat.indexOf(endMark, start + startMark.length);
  assert(start !== -1 && end > start, 'missing ' + startMark);
  return chat.slice(start, end);
}

var backend = body('async function generateWithBackend', 'function updateLlmUI');
var web = body('async function generateWithLocalLLM', 'function initSpeechRecognition');
[backend, web].forEach(function (src, idx) {
  assert(src.indexOf('content: userText') === -1, 'path ' + idx + ' still duplicates the user turn');
  assert(src.indexOf('planReply(') !== -1, 'path ' + idx + ' must plan the reply');
  assert(src.indexOf('max_tokens: budget.maxTokens') !== -1, 'path ' + idx + ' must use the budget');
  assert(src.indexOf('choiceFinishReason') !== -1 || src.indexOf('applyChoicePayload') !== -1, 'path ' + idx + ' must read finish_reason');
  assert(src.indexOf("chatLabel('chatContextFull'") !== -1, 'path ' + idx + ' must notice a full context');
});
assert(backend.indexOf(', null)') !== -1, 'Local Server passes an unknown context');
assert(chat.indexOf('function applyChoicePayload') !== -1 && chat.slice(chat.indexOf('function applyChoicePayload'), chat.indexOf('async function generateWithBackend')).indexOf('choiceFinishReason') !== -1, 'local server payload reads finish_reason');
assert((web.match(/max_tokens: budget\.maxTokens/g) || []).length >= 2, 'WebLLM streaming and non-streaming both use the budget');
assert(web.indexOf('stream: true') !== -1, 'WebLLM keeps a streaming path');
assert(web.indexOf('nonStreamReason') !== -1, 'WebLLM non-streaming reads finish_reason');
assert(web.indexOf('readLoadedContextTokens') !== -1, 'WebLLM reads the loaded context');
assert(chat.indexOf("chatLabel('chatReplyCutOff'") !== -1, 'cut-off line goes through chatLabel');

assert(en.indexOf('"chatContextFull": "Context is full. Start a new chat or remove documents to continue."') !== -1, 'en.js context-full string');
assert(en.indexOf('"chatReplyCutOff": "Reply stopped at the length limit."') !== -1, 'en.js cut-off string');

var otherPacks = fs.readdirSync(path.join(root, 'i18n')).filter(function (name) {
  return name.endsWith('.js') && name !== 'en.js';
});
otherPacks.forEach(function (name) {
  var pack = read('i18n/' + name);
  assert(pack.indexOf('chatContextFull') === -1, name + ' must not gain chatContextFull');
  assert(pack.indexOf('chatReplyCutOff') === -1, name + ' must not gain chatReplyCutOff');
});

console.log('REPLY-BUDGET-1 clamp, single user turn, notices excluded, length mark: ok');
