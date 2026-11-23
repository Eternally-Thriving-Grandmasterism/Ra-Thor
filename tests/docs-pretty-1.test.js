/* DOCS-PRETTY-1: styled public docs pages stay in step with their .md source.
 * 1. Each committed docs/*.html equals a fresh render of its .md (no hand edits, no drift).
 * 2. Wording parity, checked independently of the renderer: the words of the article text
 *    are exactly the words of the .md, in order (markup, link targets and fence tags aside).
 * 3. The raw .md stays reachable and linked; the renderer escapes HTML and drops unsafe links.
 * Contact: info@Rathor.ai */
var fs = require('fs');
var path = require('path');
var root = path.join(__dirname, '..');
var gen = require(path.join(root, 'scripts/docs-pretty/build.js'));

function assert(cond, msg) {
  if (!cond) throw new Error(msg);
}
function read(rel) {
  return fs.readFileSync(path.join(root, rel), 'utf8');
}

function words(text) {
  return (text.normalize('NFC').match(/[\p{L}\p{N}]+/gu) || []);
}

function mdWords(md) {
  var lines = md.replace(/\r\n?/g, '\n').split('\n').map(function (line) {
    if (/^\s*(`{3,}|~{3,})/.test(line)) return '';             /* fence lines and their info string */
    if (/^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*$/.test(line) && /\|/.test(line)) return ''; /* table rule */
    return line.replace(/^ {0,3}(\d{1,9}[.)])\s+/, ' ');      /* ordered-list numbers become <ol> markers */
  });
  var text = lines.join('\n').replace(/\]\(\s*<?[^()\s<>]+(?:\([^()\s]*\)[^()\s<>]*)*>?(?:\s+"[^"]*")?\s*\)/g, ']');
  return words(text);
}

function htmlWords(html) {
  var m = html.match(/<article\b[^>]*>([\s\S]*?)<\/article>/);
  assert(m, 'page has no <article>');
  var text = m[1]
    .replace(/<a class="rt-doc-anchor"[^>]*>#<\/a>/g, '')
    .replace(/<[^>]+>/g, ' ')
    .replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&quot;/g, '"').replace(/&#39;/g, "'").replace(/&amp;/g, '&');
  return words(text);
}

assert(gen.DOCS.length >= 3 && gen.DOCS.length <= 6, 'keep the styled set lean (3–6 docs)');
assert(gen.DOCS[0].md === 'docs/WHITEPAPER_v4.3.md', 'whitepaper comes first');

gen.DOCS.forEach(function (doc) {
  assert(fs.existsSync(path.join(root, doc.md)), doc.md + ' must stay at its URL');
  var html = read(doc.html);
  assert(html === gen.build(doc), doc.html + ' drifted from ' + doc.md + ' — run node scripts/docs-pretty/build.js');

  var a = mdWords(read(doc.md));
  var b = htmlWords(html);
  for (var i = 0; i < Math.max(a.length, b.length); i++) {
    if (a[i] !== b[i]) {
      throw new Error(doc.html + ' wording differs from ' + doc.md + ' at word ' + i + ': md "' +
        a.slice(Math.max(0, i - 4), i + 4).join(' ') + '" vs page "' + b.slice(Math.max(0, i - 4), i + 4).join(' ') + '"');
    }
  }

  assert(html.indexOf('href="/' + doc.md + '"') !== -1, doc.html + ' must link the raw markdown');
  if (doc.source === 'github') {
    /* WHITEPAPER-V43-1: the top-bar button opens the source on GitHub; the raw .md stays linked in the source line. */
    assert(html.indexOf('<a class="rt-doc-raw" href="https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor/blob/main/' + doc.md + '">Read the source on GitHub</a>') !== -1,
      doc.html + ' must show Read the source on GitHub');
  } else {
    assert(html.indexOf('View raw markdown') !== -1, doc.html + ' must show View raw markdown');
  }
  assert(html.indexOf('<a class="rt-doc-home" href="/">') !== -1, doc.html + ' must link home');
  assert(html.indexOf('/js/family-nav-2026-08-22.js') !== -1, doc.html + ' must carry the family nav');
  assert(html.indexOf('/js/rathor-theme.js') !== -1, doc.html + ' must carry the theme toggle');
  assert(html.indexOf('id="rt-family-main"') !== -1, doc.html + ' must give the skip link its target');
  assert(/<link rel="canonical" href="https:\/\/rathor\.ai\/docs\/[^"]+\.html">/.test(html), doc.html + ' canonical');
  var article = html.match(/<article\b[^>]*>([\s\S]*?)<\/article>/)[1];
  assert(!/<script|<iframe|<style|\son[a-z]+=|javascript:/i.test(article), doc.html + ' article must carry no active content');
  assert((article.match(/<h1\b/g) || []).length === 1, doc.html + ' must have exactly one h1');
  assert((article.match(/<table>/g) || []).length === (article.match(/class="rt-doc-table"/g) || []).length,
    doc.html + ' every table sits in a scroll wrapper');
});

/* Renderer safety. */
var evil = gen.renderMarkdown('# T\n\n<script>alert(1)</script> <img src=x onerror=alert(1)>\n\n' +
  '[a](javascript:alert(1)) [b](data:text/html,x) [c](//evil.example) [d](https://ok.example/"onmouseover=x)\n\n' +
  '| h |\n|---|\n| <b>x</b> |\n\n```\n</code><script>x</script>\n```\n', 'docs/X.md').html;
assert(evil.indexOf('<script') === -1 && evil.indexOf('<img') === -1 && evil.indexOf('<b>') === -1, 'raw HTML must be escaped');
assert(evil.indexOf('javascript:') === -1 || evil.indexOf('href="javascript:') === -1, 'javascript: links dropped');
assert(evil.indexOf('href="data:') === -1 && evil.indexOf('href="//evil') === -1, 'data: and protocol-relative links dropped');
assert(evil.indexOf('"onmouseover') === -1, 'attribute breakout escaped');

/* Link mapping. */
assert(gen.safeHref('LINEAGE.md', 'docs/CONTRIBUTIONS.md') === '/docs/LINEAGE.html', 'styled sibling links to its page');
assert(gen.safeHref('../PUBLIC_CLAIM.lock.md', 'docs/FAQ.md') ===
  'https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor/blob/main/PUBLIC_CLAIM.lock.md', 'other repo files open on GitHub');

/* Home page points readers at the styled whitepaper. */
var index = read('index.html');
assert(index.indexOf('href="/docs/WHITEPAPER_v4.3.html"') !== -1, 'index.html links the styled whitepaper');
assert(index.indexOf('href="/docs/WHITEPAPER_v4.3.md"') === -1, 'index.html no longer sends readers to raw markdown');
assert(read('js/site-lock-2026-08-22.js').indexOf("href=\"/docs/WHITEPAPER_v4.3.html\"") !== -1, 'on-device paper card links the styled whitepaper');

/* Not in the offline cache. */
['sw.js', 'public/sw.js'].forEach(function (rel) {
  if (!fs.existsSync(path.join(root, rel))) return;
  var sw = read(rel);
  gen.DOCS.forEach(function (doc) { assert(sw.indexOf(doc.html) === -1, rel + ' must not precache ' + doc.html); });
});

console.log('docs-pretty-1: ' + gen.DOCS.length + ' pages match their .md word for word');
