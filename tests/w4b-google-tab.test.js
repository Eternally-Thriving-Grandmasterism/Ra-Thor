/* W4b: Google Translate is a new-tab URL, not a widget inject. */
var fs = require('fs');
var path = require('path');
var root = path.join(__dirname, '..');
var js = fs.readFileSync(path.join(root, 'js/google-translate-optin.js'), 'utf8');
var headers = fs.readFileSync(path.join(root, '_headers'), 'utf8');
var en = fs.readFileSync(path.join(root, 'i18n/en.js'), 'utf8');

function assert(cond, msg) {
  if (!cond) throw new Error(msg);
}

assert(js.indexOf('translate_a/element.js') === -1, 'must not inject Google element.js');
assert(js.indexOf('googleTranslateElementInit') === -1, 'must not boot TranslateElement');
assert(js.indexOf('google.translate.TranslateElement') === -1, 'must not construct the widget');
assert(js.indexOf('<script') === -1, 'must not inject a translate.google.com script');
assert(js.indexOf('target="_blank"') !== -1, 'must open a new tab');
assert(js.indexOf('rel="noopener"') !== -1, 'must set rel=noopener');
assert(js.indexOf('https://rathor.ai') !== -1, 'must point Google at rathor.ai');
assert(js.indexOf('https://translate.google.com/website?sl=en&u=') !== -1, 'English must use the website target picker');
assert(js.indexOf('tl=en') === -1, 'source must never build tl=en');
assert(js.indexOf('may fail on this site (COEP)') === -1, 'must not say the proxy may fail');
assert(js.indexOf('/chat.html') !== -1, 'chat page URL must be rewritten off the chat document');
assert(en.indexOf('Opens Google Translate in a new tab. Needs the network. Not the offline pack.') !== -1, 'en note must describe the new tab');

function headerRules(text) {
  var rules = [];
  var current = null;
  text.split(/\n/).forEach(function (line) {
    var raw = line.replace(/\r$/, '');
    var trimmed = raw.trim();
    if (!trimmed || trimmed.charAt(0) === '#') return;
    if (raw.charAt(0) !== ' ' && raw.charAt(0) !== '\t') {
      current = { path: trimmed, headers: [] };
      rules.push(current);
    } else if (current) {
      current.headers.push(trimmed);
    }
  });
  return rules;
}

var rules = headerRules(headers);
var catchAll = rules.filter(function (rule) { return rule.path === '/*'; });
catchAll.forEach(function (rule) {
  rule.headers.forEach(function (header) {
    assert(header.indexOf('Cross-Origin-Embedder-Policy') === -1, 'catch-all /* must not set COEP');
    assert(header.indexOf('Cross-Origin-Opener-Policy') === -1, 'catch-all /* must not set COOP');
  });
});
assert(headers.indexOf('/*') === -1 || catchAll.length === 0 || catchAll.every(function (rule) {
  return rule.headers.every(function (header) {
    return header.indexOf('Cross-Origin-Embedder-Policy') === -1;
  });
}), '_headers must not set COEP on catch-all /*');

var chat = rules.filter(function (rule) { return rule.path === '/chat.html'; });
assert(chat.length === 1, '_headers must have one /chat.html rule');
assert(chat[0].headers.indexOf('Cross-Origin-Embedder-Policy: require-corp') !== -1, '_headers must set COEP on /chat.html');
assert(chat[0].headers.indexOf('Cross-Origin-Opener-Policy: same-origin') !== -1, '_headers must set COOP on /chat.html');
assert(headers.indexOf('unsafe-none') === -1, 'must not relax COEP on chat');

console.log('W4b google-tab checks passed');
