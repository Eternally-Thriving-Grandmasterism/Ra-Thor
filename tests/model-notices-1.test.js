/* MODEL-NOTICES-1: third-party model notices page, consent link, and licence carve-out.
 * The adopted picker table lists seven curated bases. Gemma is not a picker row. */
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

var page = read('third-party-models.html');
assert(page.indexOf('id="model-terms"') !== -1, 'page must include #model-terms');
assert(page.indexOf('id="models"') !== -1, 'page must include the picker table');

var models = page.slice(page.indexOf('id="models"'), page.indexOf('id="llama"'));
var rows = [
  'SmolLM2-360M-Instruct',
  'Qwen2.5-0.5B-Instruct',
  'Llama-3.2-1B-Instruct',
  'Qwen2.5-1.5B-Instruct',
  'Llama-3.2-3B-Instruct',
  'Phi-3.5-mini-instruct',
  'Qwen2.5-7B-Instruct'
];
var found = models.match(/<tr><td>[^<]+/g) || [];
assert(found.length === rows.length, 'picker table row count, found ' + found.length);
rows.forEach(function (name) {
  assert(models.indexOf('<td>' + name + '</td>') !== -1, 'picker table missing ' + name);
});
assert(!/gemma/i.test(models), 'no gemma row in the picker table');
assert(page.indexOf('Lattice Chat does not currently offer Gemma.') !== -1, 'Gemma stays listed as not currently offered');

var chatHtml = read('chat.html');
assert(chatHtml.indexOf('href="/third-party-models.html"') !== -1, 'chat.html must link the notices page');
assert(chatHtml.indexOf('href="/third-party-models.html#model-terms"') !== -1, 'chat.html must link #model-terms');

var footer = read('js/family-nav-2026-08-22.js');
assert(footer.indexOf("'<a href=\"/third-party-models.html\">Third-party models</a>'") !== -1, 'footer must link the notices page');

var chat = read('js/chat.js');
assert(chat.indexOf("termsNote.className = 'webllm-row-note'") !== -1, 'consent termsNote must be present');
assert(chat.indexOf("termsLink.href = '/third-party-models.html#model-terms'") !== -1, 'consent must link #model-terms');
assert(chat.indexOf('https://ai.google.dev/gemma/terms') === -1, 'gemma terms URL stays out of js/chat.js');

assert(read('LICENSE').indexOf('## 1B. Third-party components (not part of the Work)') !== -1, 'LICENSE must contain §1B');
assert(read('COMMERCIAL_LICENSE.md').indexOf('## 7A.') !== -1, 'COMMERCIAL_LICENSE.md must contain §7A');
assert(read('sitemap.xml').indexOf('<loc>https://rathor.ai/third-party-models.html</loc>') !== -1, 'sitemap must list the page');

var excepted = '(outputs of third-party models excepted; see LICENSE §1B)';
assert(read('licence.html').indexOf(excepted) !== -1, 'licence.html outputs-revenue rule must name the §1B exception');
assert(read('docs/LICENSE-COMMERCIAL.md').indexOf(excepted) !== -1, 'docs/LICENSE-COMMERCIAL.md must name the §1B exception');

var packFiles = fs.readdirSync(path.join(root, 'i18n')).filter(function (name) {
  return /^[a-z]{2}\.js$/.test(name);
}).sort();
assert(packFiles.length === 23, 'expected 23 locale packs');
packFiles.forEach(function (name) {
  var sb = { window: { translations: {} } };
  vm.createContext(sb);
  vm.runInContext(read('i18n/' + name), sb, { filename: name });
  var lang = name.slice(0, 2);
  var quote = sb.window.translations[lang] && sb.window.translations[lang].licenceFaqQuote;
  assert(typeof quote === 'string' && quote.indexOf('see LICENSE §1B') !== -1, lang + ' licenceFaqQuote must contain see LICENSE §1B');
  assert(quote.indexOf(excepted) !== -1, lang + ' licenceFaqQuote must keep the English parenthetical');
});
var enQuote = (function () {
  var sb = { window: { translations: {} } };
  vm.createContext(sb);
  vm.runInContext(read('i18n/en.js'), sb, { filename: 'en.js' });
  return sb.window.translations.en.licenceFaqQuote;
})();
assert(enQuote.indexOf('or outputs ' + excepted + ' requires') !== -1, 'English pack must place the parenthetical at outputs');

[
  'third-party/llama-3.2/LICENSE.txt',
  'third-party/llama-3.2/USE_POLICY.md',
  'third-party/llama-3.2/NOTICE.txt',
  'third-party/gemma/GEMMA_TERMS_OF_USE.txt',
  'third-party/gemma/GEMMA_PROHIBITED_USE_POLICY.txt',
  'third-party/gemma/NOTICE.txt',
  'third-party/qwen2.5/LICENSE.txt',
  'third-party/apache-2.0/LICENSE.txt',
  'third-party/phi-3.5/LICENSE.txt',
  'third-party/phi-3.5/NOTICE.md',
  'third-party/tvm/NOTICE.txt',
  'THIRD_PARTY_NOTICES.md'
].forEach(function (rel) {
  assert(fs.existsSync(path.join(root, rel)), 'missing ' + rel);
});

console.log('model-notices-1: page, seven picker rows, consent, §1B, §7A, sitemap');
