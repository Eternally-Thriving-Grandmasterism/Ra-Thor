/* RENDER-1: escape model text before markdown; Copy keeps the raw string. */
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

function tagList(html) {
  return html.match(/<[^>]+>/g) || [];
}

function assertNoExecutable(html, label) {
  assert(!/<\s*img\b/i.test(html), label + ' must not include an img tag: ' + html);
  assert(!/<\s*script\b/i.test(html), label + ' must not include a script tag: ' + html);
  assert(!/<\s*iframe\b/i.test(html), label + ' must not include an iframe: ' + html);
  assert(html.indexOf('<!--') === -1, label + ' must not open an HTML comment: ' + html);
  tagList(html).forEach(function (tag) {
    assert(!/\son[a-z]+\s*=/i.test(tag), label + ' handler in ' + tag);
    assert(!/javascript:/i.test(tag), label + ' javascript in ' + tag);
    assert(!/\bdata:/i.test(tag), label + ' data URL in ' + tag);
  });
}

var chat = read('js/chat.js');
var sw = read('sw.js');

assert(chat.indexOf('<think>') === -1, 'chat.js must not gain a think tag');
assert(chat.indexOf('window.confirm') === -1, 'chat.js must not call window.confirm');
assert(chat.indexOf("./vendor/web-llm/0.2.85/index.js") !== -1, 'WebLLM pin stays 0.2.85');
assert(chat.indexOf('max_tokens: 500') !== -1, 'WebLLM max_tokens stays 500');
assert(chat.indexOf('max_tokens: 900') !== -1, 'Local Server max_tokens stays 900');
assert(sw.indexOf("var LOCK = '20260924a';") !== -1, 'service worker LOCK stays 20260924a');

var promptMarker = 'const SYSTEM_PROMPT = `';
var promptAt = chat.indexOf(promptMarker);
var promptEnd = chat.indexOf('`;', promptAt);
var prompt = chat.slice(promptAt + promptMarker.length, promptEnd);
var promptHash = crypto.createHash('sha256').update(prompt).digest('hex');
assert(promptHash === '1fc78b6d442e43494de3bb52adc5307eb3c7861476cd0d838af0c3fcd61def61', 'SYSTEM_PROMPT hash changed: ' + promptHash);

var pureStart = chat.indexOf('/* chat-render-pure */');
var pureEnd = chat.indexOf('/* chat-render-pure-end */');
assert(pureStart !== -1 && pureEnd > pureStart, 'render pure block must be marked');
var pure = chat.slice(pureStart, pureEnd);
assert(pure.indexOf('function escapeHtml') !== -1, 'escape helper lives in the pure block');
assert(pure.indexOf('function renderText') !== -1, 'renderText lives in the pure block');
assert(pure.indexOf('escapeHtml(text)') !== -1, 'renderText must escape before markdown');
assert(pure.indexOf(".replace(/&/g, '&amp;')") !== -1, 'ampersand is escaped');
assert(pure.indexOf(".replace(/</g, '&lt;')") !== -1, 'less-than is escaped');
assert(pure.indexOf(".replace(/>/g, '&gt;')") !== -1, 'greater-than is escaped');
assert(pure.indexOf(".replace(/\"/g, '&quot;')") !== -1, 'double quote is escaped');
assert(pure.indexOf(".replace(/'/g, '&#39;')") !== -1, 'single quote is escaped');

var sandbox = {};
vm.createContext(sandbox);
vm.runInContext(pure + '\nthis.api = { renderText: renderText, escapeHtml: escapeHtml };', sandbox);
var api = sandbox.api;

var lessThan = api.renderText('x<y and more');
assert(lessThan === 'x&lt;y and more', 'x<y keeps the rest: ' + lessThan);
assert(lessThan.indexOf('and more') !== -1, 'text after < stays');
assertNoExecutable(lessThan, 'x<y');

var comment = api.renderText('<!-- tail');
assert(comment.indexOf('tail') !== -1, 'comment payload still shows tail');
assert(comment.indexOf('<!--') === -1, 'comment opener is escaped');
assert(comment === '&lt;!-- tail', 'comment renders as text: ' + comment);

var bare = api.renderText('alpha < beta remains visible');
assert(bare.indexOf('beta remains visible') !== -1, 'text after a bare < stays');
assert(bare === 'alpha &lt; beta remains visible', 'bare < is escaped: ' + bare);
assertNoExecutable(bare, 'bare <');

var img = api.renderText('<img src=x onerror=alert(1)>');
assert(img.indexOf('<img') === -1, 'img payload must not yield an img tag: ' + img);
assert(img.indexOf('&lt;img') !== -1, 'img payload stays visible as text');
assertNoExecutable(img, 'img payload');

var fenced = api.renderText('```\n<script>alert(1)</script>\n```');
assert(fenced.indexOf('<pre>') !== -1, 'code fence still wraps pre: ' + fenced);
assert(fenced.indexOf('<code>') !== -1, 'code fence still wraps code: ' + fenced);
assert(fenced.indexOf('&lt;script&gt;') !== -1, 'script inside a fence is escaped text');
assert(fenced.indexOf('<script') === -1, 'fence must not emit a script tag');
assert(fenced.indexOf('</script>') === -1, 'fence must not emit a closing script tag');
assertNoExecutable(fenced, 'fenced script');

var jsLink = api.renderText('[a](javascript:alert(1))');
assert(!/href\s*=\s*['"]?\s*javascript:/i.test(jsLink), 'javascript: link must not become an href: ' + jsLink);
assert(jsLink.indexOf('<a ') === -1, 'javascript: link stays plain text: ' + jsLink);
assertNoExecutable(jsLink, 'javascript link');

var dataLink = api.renderText('[a](data:text/html,hi)');
assert(dataLink.indexOf('<a ') === -1, 'data: link stays plain text: ' + dataLink);
assertNoExecutable(dataLink, 'data link');

var httpsLink = api.renderText('[a](https://b.example)');
assert(httpsLink.indexOf('<a ') !== -1, 'https link still renders: ' + httpsLink);
assert(httpsLink.indexOf('href="https://b.example"') !== -1, 'https href stays: ' + httpsLink);
assert(httpsLink.indexOf('target="_blank"') !== -1, 'https link opens a new tab');
assert(httpsLink.indexOf('rel="noopener noreferrer"') !== -1, 'https link sets noopener noreferrer');
assertNoExecutable(httpsLink, 'https link');

var quoted = api.renderText('[a](https://b.example/x"y)');
assert(quoted.indexOf('href="https://b.example/x&quot;y"') !== -1, 'href is quote-safe: ' + quoted);
assert(quoted.indexOf('x"y') === -1, 'raw quote must not break the href');
assertNoExecutable(quoted, 'quoted href');

var md = api.renderText('# Title\n\n- item\n\n**bold**');
assert(md.indexOf('<h1>Title</h1>') !== -1, 'header still renders: ' + md);
assert(md.indexOf('<ul><li>item</li></ul>') !== -1, 'list still renders: ' + md);
assert(md.indexOf('<strong>bold</strong>') !== -1, 'bold still renders: ' + md);
assertNoExecutable(md, 'markdown');

var saved = [
  { role: 'rathor', text: '<img src=x onerror=alert(1)>' },
  { role: 'user', text: 'before <!-- tail after' }
];
saved.forEach(function (entry) {
  var again = api.renderText(entry.text);
  assertNoExecutable(again, 're-rendered history');
  if (entry.text.indexOf('tail') !== -1) {
    assert(again.indexOf('tail') !== -1, 're-rendered history keeps tail');
  }
  if (entry.text.indexOf('<img') !== -1) {
    assert(again.indexOf('<img') === -1, 're-rendered history has no img tag');
  }
});

var fileName = '<img src=x onerror=1>.txt';
var escapedName = api.escapeHtml(fileName);
assert(escapedName === '&lt;img src=x onerror=1&gt;.txt', 'docs name escape: ' + escapedName);
assert(escapedName.indexOf('<img') === -1, 'escaped file name has no img tag');
assert(api.escapeHtml('a&b "c" \'d\'') === 'a&amp;b &quot;c&quot; &#39;d&#39;', 'escape helper covers quotes');

var docs = chat.slice(chat.indexOf('function renderDocsBar'), chat.indexOf('function handleDocumentUpload'));
assert(docs.indexOf('escapeHtml(d.name)') !== -1, 'docs bar escapes the file name');
assert(docs.indexOf('escapeHtml(d.id)') !== -1, 'docs bar escapes the file id');

var add = chat.slice(chat.indexOf('function addMessage('), chat.indexOf('function finalizeStreamingMessage('));
assert(add.indexOf('msgDiv.rawText = text') !== -1, 'addMessage stores raw text on the message');
assert(add.indexOf('escapeHtml(relativeTime(timestamp))') !== -1, 'message meta escapes interpolated text');
assert(add.indexOf('innerText') === -1, 'Copy must not read innerText');
assert(add.indexOf('msgDiv.rawText') !== -1, 'Copy reads the stored raw text');
assert(add.indexOf('renderText(text)') !== -1, 'addMessage renders through renderText');

var fin = chat.slice(chat.indexOf('function finalizeStreamingMessage('), chat.indexOf('function renderHistory('));
assert(fin.indexOf('msgDiv.rawText = finalText') !== -1, 'finalize updates the stored raw text');

var hist = chat.slice(chat.indexOf('function renderHistory('), chat.indexOf('function updateSessionMeta('));
assert(hist.indexOf('addMessage(m.text, m.role, false, m.ts)') !== -1, 'history re-renders through addMessage');

assert(chat.indexOf('textDiv.innerText') === -1, 'Copy no longer uses innerText');

console.log('RENDER-1 renderText escape, safe links, raw copy, docs escape: ok');
