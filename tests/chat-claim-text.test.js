/* Claim text: the two softened chat-privacy phrases stay out of the named files. */
var fs = require('fs');
var path = require('path');

var root = path.join(__dirname, '..');

function assert(cond, msg) {
  if (!cond) throw new Error(msg);
}

function read(rel) {
  return fs.readFileSync(path.join(root, rel), 'utf8');
}

['i18n/en.js', 'chat.html', 'privacy.html'].forEach(function (rel) {
  var text = read(rel);
  assert(text.indexOf('for maximum privacy') === -1, rel + ' still says for maximum privacy');
  assert(text.indexOf('Zero personal data leaves your browser') === -1, rel + ' still says Zero personal data leaves your browser');
});

var en = read('i18n/en.js');
assert(en.indexOf('Chats are saved only in this browser. If you connect a Local Server or an online provider, your messages are sent there.') !== -1, 'English chatReplyPrivacy uses the approved wording');
assert(en.indexOf('via the lock button. Forgetting the passphrase makes the data unrecoverable.') !== -1, 'the lock sentence stays, without the maximum-privacy clause');

console.log('chat-claim-text ok');
