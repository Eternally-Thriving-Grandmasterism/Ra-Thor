/* CLAIM-WHO-1: chatReplyWho names where replies are made. */
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

function loadPack(lang) {
  var sandbox = { window: { translations: {} } };
  vm.createContext(sandbox);
  vm.runInContext(read('i18n/' + lang + '.js'), sandbox, { filename: lang + '.js' });
  var pack = sandbox.window.translations[lang];
  assert(pack && typeof pack === 'object', lang + ' pack must assign window.translations.' + lang);
  return pack;
}

var NEW_SENTENCE = 'Responses are made in this browser unless you connect a Local Server or an online provider, which then receives your messages.';

var en = loadPack('en');
assert(typeof en.chatReplyWho === 'string', 'en chatReplyWho must stay defined');
assert(en.chatReplyWho.indexOf('All responses stay on your device') === -1, 'en chatReplyWho still says All responses stay on your device');
assert(en.chatReplyWho.indexOf(NEW_SENTENCE) !== -1, 'en chatReplyWho must use the Steward-approved sentence');
assert(en.chatReplyWho.indexOf('No data is collected.') !== -1, 'en chatReplyWho must keep No data is collected.');

/* Packs that defined chatReplyWho at tip 87659808. A pack added later is out of this list. */
var HAD_WHO = [
  'ar', 'de', 'el', 'en', 'es', 'fa', 'fr', 'he', 'hi', 'id', 'it', 'ja',
  'ko', 'nl', 'pl', 'pt', 'ru', 'sv', 'th', 'tr', 'uk', 'vi', 'zh'
];

HAD_WHO.forEach(function (lang) {
  var pack = loadPack(lang);
  assert(typeof pack.chatReplyWho === 'string' && pack.chatReplyWho.trim() !== '', lang + ' must still define chatReplyWho');
});

console.log('chat-claim-who ok');
