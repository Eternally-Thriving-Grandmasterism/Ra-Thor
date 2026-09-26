/* CLAIM-DOC-VOICE-1: chatReplyDoc and the voice line name where document text and speech go.
   CLAIM-TTS-1: the TTS label must not call read-aloud offline. */
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

var NEW_DOC = 'Document text stays in this browser unless you connect a Local Server or an online provider, which then receives it with your messages.';
var OLD_DOC = 'Everything stays on your device.';
var DOC_PREFIX = 'Use the document button (file icon) next to the mic to upload .txt, .md, .json or .csv files. Their content is injected into the conversation context for Local Server / WebLLM / Copy Context. ';
var NEW_VOICE = "Uses your browser's built-in speech service. Some browsers, such as Chrome, send your voice to their own servers to recognize it.";
var NEW_TTS_SENTENCE = 'Some read-aloud voices are online too, and receive the text they read.';
var NEW_TTS_LABEL = 'Enable read-aloud (TTS)';
var OLD_TTS_LABEL = 'Enable offline speech';

var en = loadPack('en');
assert(typeof en.chatReplyDoc === 'string', 'en chatReplyDoc must stay defined');
assert(en.chatReplyDoc.indexOf(OLD_DOC) === -1, 'en chatReplyDoc still says Everything stays on your device.');
assert(en.chatReplyDoc.indexOf(NEW_DOC) !== -1, 'en chatReplyDoc must use the Steward-approved sentence');
assert(en.chatReplyDoc.indexOf(DOC_PREFIX) === 0, 'en chatReplyDoc must keep the upload instructions');
assert(en.chatReplyWho.indexOf('No data is collected.') !== -1, 'en chatReplyWho must keep No data is collected.');

/* Packs that defined chatReplyDoc at tip 26e9e1af. A pack added later is out of this list. */
var HAD_DOC = [
  'ar', 'de', 'el', 'en', 'es', 'fa', 'fr', 'he', 'hi', 'id', 'it', 'ja',
  'ko', 'nl', 'pl', 'pt', 'ru', 'sv', 'th', 'tr', 'uk', 'vi', 'zh'
];

/* Last sentence of chatReplyDoc at tip 26e9e1af. chatReplyHelp uses a different closing. */
var OLD_CLOSINGS = {
  ar: 'كل شيء يبقى على جهازك.',
  de: 'Alles bleibt auf Ihrem Gerät.',
  el: 'Όλα μένουν στη συσκευή σας.',
  en: OLD_DOC,
  es: 'Todo queda en tu dispositivo.',
  fa: 'همه چیز روی دستگاه شما می‌ماند.',
  fr: 'Tout reste sur votre appareil.',
  he: 'הכול נשאר במכשיר.',
  hi: 'सब आपके उपकरण पर रहता है।',
  id: 'Semuanya tinggal di perangkat Anda.',
  it: 'Tutto resta sul tuo dispositivo.',
  ja: 'すべては機器に残ります。',
  ko: '모든 것은 기기에 남습니다.',
  nl: 'Alles blijft op uw apparaat.',
  pl: 'Wszystko zostaje na urządzeniu.',
  pt: 'Tudo fica no seu dispositivo.',
  ru: 'Всё остаётся на устройстве.',
  sv: 'Allt stannar på din enhet.',
  th: 'ทุกอย่างอยู่บนอุปกรณ์ของคุณ',
  tr: 'Her şey cihazınızda kalır.',
  uk: 'Усе лишається на пристрої.',
  vi: 'Mọi thứ ở lại trên thiết bị.',
  zh: '一切留在你的设备上。'
};

HAD_DOC.forEach(function (lang) {
  var pack = loadPack(lang);
  assert(typeof pack.chatReplyDoc === 'string' && pack.chatReplyDoc.trim() !== '', lang + ' must still define chatReplyDoc');
  assert(pack.chatReplyDoc.indexOf(OLD_DOC) === -1, lang + ' chatReplyDoc still has the English on-device sentence');
  assert(pack.chatReplyDoc.indexOf(OLD_CLOSINGS[lang]) === -1, lang + ' chatReplyDoc still has its old on-device closing');
});

var html = read('chat.html');
var voiceStart = html.indexOf('id="voice-settings-overlay"');
var voiceEnd = html.indexOf('id="unlock-overlay"');
assert(voiceStart !== -1 && voiceEnd > voiceStart, 'voice settings block must exist');
var voiceBlock = html.slice(voiceStart, voiceEnd);
assert(voiceBlock.indexOf('Stays on your device.') === -1, 'voice block still says Stays on your device.');
assert(voiceBlock.indexOf(NEW_VOICE) !== -1, 'voice block must use the corrected speech line');
assert(voiceBlock.indexOf(NEW_VOICE + ' ' + NEW_TTS_SENTENCE) !== -1, 'read-aloud sentence must follow the recognition sentence');
assert(voiceBlock.indexOf(NEW_TTS_LABEL) !== -1, 'voice block must use Enable read-aloud (TTS)');
assert(voiceBlock.indexOf(OLD_TTS_LABEL) === -1, 'voice block still says Enable offline speech');
assert(voiceBlock.indexOf('data-i18n') === -1, 'voice block stays hardcoded; it had no data-i18n pattern');

console.log('chat-claim-doc-voice ok');
