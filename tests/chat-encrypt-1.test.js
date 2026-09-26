/* ENCRYPT-FIX-1: passphrase lock stays an envelope on every save. */
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
var en = read('i18n/en.js');

assert(chat.indexOf('<think>') === -1, 'chat.js must not gain a think tag');
assert(chat.indexOf('window.confirm') === -1, 'chat.js must not call window.confirm');
assert(chat.indexOf('prompt(') !== -1, 'prompt() stays for this card');
assert(chat.indexOf('while unlocked we keep a temporary plain write') === -1, 'plaintext-while-unlocked comment must be gone');
assert(chat.indexOf('removeItem(ENCRYPT_FLAG)') === -1, 'flag is cleared only by a turn-off control, and this card has none');

var promptMarker = 'const SYSTEM_PROMPT = `';
var promptAt = chat.indexOf(promptMarker);
var promptEnd = chat.indexOf('`;', promptAt);
var prompt = chat.slice(promptAt + promptMarker.length, promptEnd);
var promptHash = crypto.createHash('sha256').update(prompt).digest('hex');
assert(promptHash === '1fc78b6d442e43494de3bb52adc5307eb3c7861476cd0d838af0c3fcd61def61', 'SYSTEM_PROMPT hash changed: ' + promptHash);

var pureStart = chat.indexOf('/* chat-encrypt-pure */');
var pureEnd = chat.indexOf('/* chat-encrypt-pure-end */');
assert(pureStart !== -1 && pureEnd > pureStart, 'encrypt pure block must be marked');
var pure = chat.slice(pureStart, pureEnd);
assert(pure.indexOf('iterations: 100000') !== -1, 'PBKDF2 rounds stay 100000');
assert(pure.split('iterations:').length === 2, 'PBKDF2 rounds are declared once');
assert(pure.indexOf("hash: 'SHA-256'") !== -1, 'PBKDF2 hash stays SHA-256');
assert(pure.indexOf("name: 'AES-GCM'") !== -1, 'cipher stays AES-GCM');
assert(pure.indexOf('length: 256') !== -1, 'AES-GCM length stays 256');
assert(pure.indexOf('new Uint8Array(16)') !== -1, 'salt stays 16 bytes');
assert(pure.indexOf('new Uint8Array(12)') !== -1, 'IV stays 12 bytes');
assert(pure.indexOf('version: 1') !== -1, 'envelope version stays 1');
assert(pure.indexOf('localStorage') === -1, 'pure helpers must not touch localStorage');

var saveStart = chat.indexOf('function saveStore()');
var saveEnd = chat.indexOf('async function enableEncryption');
assert(saveStart !== -1 && saveEnd > saveStart, 'saveStore must sit ahead of enableEncryption');
var saveFn = chat.slice(saveStart, saveEnd);
assert(saveFn.indexOf('payloadForSave') !== -1, 'saveStore must seal through payloadForSave');
assert(saveFn.indexOf('JSON.stringify(store)') === -1, 'saveStore must not write the plain store itself');
var refuseAt = saveFn.indexOf('refuseSaveOverEnvelope');
var writeAt = saveFn.indexOf('localStorage.setItem(STORE_KEY');
assert(refuseAt !== -1 && writeAt > refuseAt, 'an envelope on disk is checked before any write');

var enableStart = chat.indexOf('async function enableEncryption');
var enableEnd = chat.indexOf('async function tryUnlock');
assert(enableStart !== -1 && enableEnd > enableStart, 'enableEncryption must exist');
var enableFn = chat.slice(enableStart, enableEnd);
assert(enableFn.indexOf('beginPassphraseLock') !== -1, 'enable must keep a held key via beginPassphraseLock');
assert(enableFn.indexOf('cryptoKey = null') === -1, 'enable must not drop the held key');
assert(enableFn.indexOf('localStorage.setItem(ENCRYPT_FLAG, begun.flag)') !== -1, 'enable must write ENCRYPT_FLAG');
assert(enableFn.indexOf('prompt(') !== -1, 'enable still asks with prompt()');
assert(enableFn.indexOf('addMessage(') !== -1, 'confirm still goes through addMessage after the key is held');
var keySet = enableFn.indexOf('cryptoKey = begun.cryptoKey');
var confirmAt = enableFn.indexOf('Session store is now encrypted');
assert(keySet !== -1 && confirmAt > keySet, 'the held key is set before the confirm message');

var unlockStart = chat.indexOf('async function tryUnlock');
var unlockEnd = chat.indexOf('function warnIfPlaintextUnderFlag');
assert(unlockStart !== -1 && unlockEnd > unlockStart, 'tryUnlock must exist');
var unlockFn = chat.slice(unlockStart, unlockEnd);
assert(unlockFn.indexOf('isEncrypted = false') === -1, 'unlock must keep the store encrypted');
assert(unlockFn.indexOf('loadStore(pass)') !== -1, 'unlock still loads with the passphrase');
assert(unlockFn.indexOf('flagAfterUnlock') !== -1, 'unlock sets the flag for an older envelope');
assert(unlockFn.indexOf('localStorage.setItem(ENCRYPT_FLAG, nextFlag)') !== -1, 'unlock writes ENCRYPT_FLAG');
assert(unlockFn.indexOf('setLockedAppInert(false)') !== -1, 'unlock removes inert from the app');
var loadFn = chat.slice(chat.indexOf('async function loadStore'), chat.indexOf('function createDefaultSession'));
var failAt = loadFn.indexOf("console.warn('[Ra-Thor] decrypt failed'");
assert(failAt !== -1, 'a wrong passphrase still fails inside loadStore');
var failSlice = loadFn.slice(failAt, loadFn.indexOf('return false;', failAt));
assert(failSlice.indexOf('isEncrypted') === -1, 'a wrong passphrase must not reset isEncrypted');

var warn = chat.slice(unlockEnd, chat.indexOf('function uid()'));
assert(warn.indexOf("chatLabel('chatEncryptPlainNotice'") !== -1, 'notice goes through chatLabel');
assert(warn.indexOf('Your saved chats are not encrypted right now. Set your passphrase again to lock them.') !== -1, 'notice fallback wording');
assert(warn.indexOf(", false);") !== -1, 'notice is not persisted');
assert(warn.indexOf('saveStore') === -1, 'notice function must not save');

var init = chat.slice(chat.indexOf("window.addEventListener('DOMContentLoaded'"));
var encBranch = init.indexOf('if (isStoreEncrypted())');
var loadAt = init.indexOf('await loadStore()');
var warnAt = init.indexOf('warnIfPlaintextUnderFlag()');
assert(encBranch !== -1 && loadAt > encBranch && warnAt > loadAt, 'plaintext notice runs only after a plain load');
var lockedLoad = init.slice(encBranch, loadAt);
assert(lockedLoad.indexOf('isEncrypted = true') !== -1, 'a locked load sets isEncrypted');
assert(lockedLoad.indexOf('cryptoKey = null') !== -1, 'a locked load leaves the key null');
assert(lockedLoad.indexOf('cryptoSalt = null') !== -1, 'a locked load leaves the salt null');
assert(lockedLoad.indexOf('setLockedAppInert(true)') !== -1, 'the app behind the passphrase screen is inert');
var encSet = lockedLoad.indexOf('isEncrypted = true');
var encReturn = lockedLoad.indexOf('return;');
assert(encSet !== -1 && encReturn > encSet, 'isEncrypted is set before the locked return');

assert(en.indexOf('"chatEncryptPlainNotice": "Your saved chats are not encrypted right now. Set your passphrase again to lock them."') !== -1, 'en.js notice key');
assert(en.indexOf('Zero personal data leaves your browser') === -1, 'en.js drops the zero-personal-data browser claim');
assert(en.indexOf('for maximum privacy') === -1, 'en.js drops for maximum privacy');
assert(en.indexOf('Chats are saved only in this browser. If you connect a Local Server or an online provider, your messages are sent there.') !== -1, 'chat privacy reply uses the approved wording');

var sandbox = {
  crypto: globalThis.crypto,
  btoa: btoa,
  atob: atob,
  TextEncoder: TextEncoder,
  TextDecoder: TextDecoder,
  Uint8Array: Uint8Array,
  JSON: JSON,
  console: console
};
vm.createContext(sandbox);
vm.runInContext(
  pure + '\nthis.api = { encryptStore: encryptStore, decryptStore: decryptStore, beginPassphraseLock: beginPassphraseLock, openLockedEnvelope: openLockedEnvelope, payloadForSave: payloadForSave, plaintextDespiteFlag: plaintextDespiteFlag, encryptionFlagValue: encryptionFlagValue, base64ToBuf: base64ToBuf, refuseSaveOverEnvelope: refuseSaveOverEnvelope, flagAfterUnlock: flagAfterUnlock };',
  sandbox
);
var api = sandbox.api;

var SECRET = 'ENCRYPT-FIX-1-ROUNDTRIP-SECRET';
var AFTER = 'AFTER-UNLOCK-SECRET';
var pass = 'correct-horse';
var storeObj = {
  activeId: 's1',
  sessions: {
    s1: {
      id: 's1',
      name: 'Session 1',
      history: [{ role: 'user', text: SECRET }]
    }
  }
};

var sealed = null;

function main() {
  return api.encryptStore(pass, storeObj).then(function (envelope) {
    sealed = envelope;
    assert(envelope.encrypted === true, 'encryptStore sets encrypted');
    assert(envelope.version === 1, 'encryptStore version is 1');
    assert(api.base64ToBuf(envelope.iv).length === 12, 'encryptStore IV is 12 bytes');
    assert(api.base64ToBuf(envelope.salt).length === 16, 'encryptStore salt is 16 bytes');
    assert(JSON.stringify(envelope).indexOf(SECRET) === -1, 'round-trip envelope hides the secret');
    return api.decryptStore(pass, envelope).then(function (opened) {
      assert(opened.sessions.s1.history[0].text === SECRET, 'decryptStore restores the secret');
      return api.openLockedEnvelope(pass, envelope).then(function (held) {
        assert(held.isEncrypted === true, 'unlock keeps isEncrypted true');
        assert(held.cryptoKey, 'unlock keeps the AES-GCM key');
        assert(held.cryptoSalt && held.cryptoSalt.length === 16, 'unlock keeps the salt');
        var next = JSON.parse(JSON.stringify(held.store));
        next.sessions.s1.history.push({ role: 'user', text: AFTER });
        return api.payloadForSave(
          { isEncrypted: held.isEncrypted, cryptoKey: held.cryptoKey, cryptoSalt: held.cryptoSalt },
          next
        ).then(function (saved) {
          assert(saved.plaintext === false, 'save after unlock is not plaintext');
          assert(saved.refused === false, 'save after unlock is not refused');
          assert(saved.body.indexOf(SECRET) === -1, 'saved envelope hides the original secret');
          assert(saved.body.indexOf(AFTER) === -1, 'saved envelope hides the post-unlock secret');
          assert(saved.body !== JSON.stringify(next), 'saved body is not the plain store');
          var parsed = JSON.parse(saved.body);
          assert(parsed.encrypted === true, 'saved body is an envelope');
          assert(parsed.version === 1, 'saved envelope version is 1');
          assert(parsed.salt === envelope.salt, 'save after unlock reuses the salt');
          assert(parsed.iv !== envelope.iv, 'save after unlock uses a new IV');
          assert(api.base64ToBuf(parsed.iv).length === 12, 'saved IV is 12 bytes');
          return api.payloadForSave(
            { isEncrypted: held.isEncrypted, cryptoKey: held.cryptoKey, cryptoSalt: held.cryptoSalt },
            next
          ).then(function (saved2) {
            var parsed2 = JSON.parse(saved2.body);
            assert(parsed2.salt === parsed.salt, 'second save keeps the same salt');
            assert(parsed2.iv !== parsed.iv, 'second save uses another IV');
            assert(parsed2.version === 1 && parsed2.encrypted === true, 'second save stays version 1');
            return api.decryptStore(pass, parsed).then(function (round) {
              assert(round.sessions.s1.history[0].text === SECRET, 'envelope from save-after-unlock still opens');
              assert(round.sessions.s1.history[1].text === AFTER, 'post-unlock message survives the envelope');
              return api.decryptStore('wrong-passphrase', parsed).then(function () {
                throw new Error('wrong passphrase must fail');
              }, function () {
                return api.decryptStore(pass, envelope);
              });
            }).then(function (still) {
              assert(still.sessions.s1.history[0].text === SECRET, 'a wrong passphrase leaves the original envelope readable');
              return api.openLockedEnvelope('wrong-passphrase', envelope).then(function () {
                throw new Error('openLockedEnvelope must fail closed');
              }, function () {
                return api.payloadForSave({ isEncrypted: false, cryptoKey: null, cryptoSalt: null }, storeObj);
              });
            });
          });
        });
      });
    });
  }).then(function (plain) {
    assert(plain.plaintext === true, 'no lock writes plaintext');
    assert(plain.body.indexOf(SECRET) !== -1, 'plaintext save still contains the secret');
    return api.payloadForSave({ isEncrypted: true, cryptoKey: null, cryptoSalt: null }, storeObj);
  }).then(function (refused) {
    assert(refused.refused === true && refused.body == null && refused.plaintext === false, 'a lock without a key must not write plaintext');
    var lockedRaw = JSON.stringify(sealed);
    assert(api.refuseSaveOverEnvelope({ cryptoKey: null, isEncrypted: false }, lockedRaw) === true, 'a save while locked (encrypted store, no key) must not write');
    var kept = lockedRaw;
    if (!api.refuseSaveOverEnvelope({ cryptoKey: null, isEncrypted: false }, lockedRaw)) kept = JSON.stringify(storeObj);
    assert(kept === lockedRaw, 'the refused save leaves the envelope in place');
    assert(kept.indexOf('"sessions"') === -1, 'the refused save does not replace the envelope with a session store');
    assert(api.refuseSaveOverEnvelope({ cryptoKey: null }, JSON.stringify(storeObj)) === false, 'a plain store is not blocked by the envelope backstop');
    assert(api.flagAfterUnlock(null) === '1', 'unlock of an older envelope sets ENCRYPT_FLAG');
    assert(api.flagAfterUnlock('') === '1', 'an empty flag becomes 1 after unlock');
    assert(api.flagAfterUnlock('1') === '1', 'an existing flag stays 1');
    assert(api.encryptionFlagValue() === '1', 'flag value is 1');
    assert(api.plaintextDespiteFlag('1', JSON.stringify(storeObj)) === true, 'flag plus plaintext store asks for a notice');
    assert(api.plaintextDespiteFlag('1', JSON.stringify({ encrypted: true, version: 1, salt: 'aa', iv: 'bb', data: 'cc' })) === false, 'flag plus envelope is not a plaintext notice');
    assert(api.plaintextDespiteFlag(null, JSON.stringify(storeObj)) === false, 'plaintext without the flag is quiet');
    assert(api.plaintextDespiteFlag('1', '') === false, 'empty storage is quiet');
    return api.beginPassphraseLock(pass, storeObj);
  }).then(function (begun) {
    assert(begun.flag === '1', 'enable writes flag 1');
    assert(begun.isEncrypted === true, 'enable leaves the lock on');
    assert(begun.body.indexOf(SECRET) === -1, 'enable body hides the secret');
    var env = JSON.parse(begun.body);
    assert(env.encrypted === true && env.version === 1, 'enable body is a version 1 envelope');
    assert(env.salt === begun.envelope.salt, 'enable body uses the held salt');
    return api.decryptStore(pass, env).then(function (back) {
      assert(back.sessions.s1.history[0].text === SECRET, 'enable envelope opens with the same passphrase');
    });
  });
}

main().then(function () {
  console.log('chat-encrypt-1 ok');
}).catch(function (err) {
  console.error(err);
  process.exit(1);
});
