/* MERCY-CHECKS-WORDING-3: word-bounded LOCAL_KNOWLEDGE routing to chatReplyTolc. */
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
var expected =
  "{ q: /\\b(?:tolc\\s?8?|mercy[- ]?check(?:s|ing|ed)?|mercy[- ]?ways?|mercy[- ]?gates?|gates?|ethics|guardrails)\\b/i, k: 'chatReplyTolc' }";
assert(chat.indexOf(expected) !== -1, 'chatReplyTolc regex must stay word-bounded');

var promptMarker = 'const SYSTEM_PROMPT = `';
var promptAt = chat.indexOf(promptMarker);
var promptEnd = chat.indexOf('`;', promptAt);
var prompt = chat.slice(promptAt + promptMarker.length, promptEnd);
var promptHash = crypto.createHash('sha256').update(prompt).digest('hex');
assert(
  promptHash === 'b4fe56c33d663249f58a6a99de1ff0b2c3131218805b9ffa67faee2dcc65b4f0',
  'SYSTEM_PROMPT hash changed: ' + promptHash
);

var line = chat.split('\n').filter(function (l) {
  return l.indexOf("k: 'chatReplyTolc'") !== -1;
})[0];
var reSrc = line.match(/q: (\/\\b.*?\/i)/)[1];
var tolRegex = Function('return ' + reSrc)();

var positives = [
  'tolc',
  'TOLC 8',
  'TOLC8',
  'tolc8',
  'mercy check',
  'mercy checks',
  'Mercy-Checks',
  'Mercy-Checking',
  'Mercy-Checked',
  'Mercy-Ways',
  'the mercy ways',
  'mercy checking account',
  'mercy gate',
  'gates',
  'gate',
  'ethics',
  'guardrails'
];
positives.forEach(function (text) {
  assert(tolRegex.test(text), 'expected match: ' + text);
});

var negatives = ['mercy checkout', 'mercy wayside', 'tolcfoo'];
negatives.forEach(function (text) {
  assert(!tolRegex.test(text), 'expected no match: ' + text);
});

console.log('chat-tolc-routing ok');
