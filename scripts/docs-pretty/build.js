#!/usr/bin/env node
/* DOCS-PRETTY-1 — render key public docs as styled static pages in the site's own look.
 *
 * Source of truth stays the .md file. Each page is generated from it and committed.
 *   node scripts/docs-pretty/build.js          write the pages
 *   node scripts/docs-pretty/build.js --check  exit 1 if a committed page drifts from its .md
 *
 * No dependencies. The renderer escapes all text and attributes and drops raw HTML,
 * so a doc can never inject markup or script into its page. Link schemes are
 * allow-listed (http, https, mailto, relative, #anchor).
 * Contact: info@Rathor.ai
 */
'use strict';
var fs = require('fs');
var path = require('path');

var ROOT = path.join(__dirname, '..', '..');
var REPO_BLOB = 'https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor/blob/main/';
var SITE = 'https://rathor.ai/';

/* The docs that get a styled page. Visitor-voice docs only: operator spines that carry
 * HOLD lines (docs/EMPLOY.md, docs/ADOPT.md) stay raw, per the public / operator voice split.
 * md: repo path of the source. html: repo path of the page.
 * source: 'github' makes the top-bar button read "Read the source on GitHub" and open the .md on GitHub;
 * otherwise it reads "View raw markdown" and opens the raw .md on rathor.ai.
 * WHITEPAPER-V43-1: v4.3 is the current paper. docs/WHITEPAPER_v4.2.html stays at its URL as a frozen,
 * byte-identical history page and is no longer regenerated here. */
var DOCS = [
  { md: 'docs/WHITEPAPER_v4.3.md', html: 'docs/WHITEPAPER_v4.3.html', source: 'github' },
  { md: 'docs/FAQ.md', html: 'docs/FAQ.html' },
  { md: 'docs/LINEAGE.md', html: 'docs/LINEAGE.html' },
  { md: 'docs/CONTRIBUTIONS.md', html: 'docs/CONTRIBUTIONS.html' }
];

/* ART-UPLIFT-1: one shared doc cover scene; each page differs by crop and tint only.
 * Sits between .rt-doc-bar and .rt-doc-layout, outside the article, so wording parity holds. */
var COVER = '/assets/art/' + 'hero-docs-archive-light'; /* split so only built pages carry the full URL */
var COVER_ALT = 'Ra-Thor in dark gold-trimmed armour with a winged helm and an emerald eye seal on the shoulder reads a glowing open book in a candlelit library as loose pages lift on threads of blue lightning.';
var COVER_VARIANT = {
  'docs/WHITEPAPER_v4.3.html': '',
  'docs/FAQ.html': ' rt-art-hero-wide rt-art-hero-warm',
  'docs/LINEAGE.html': ' rt-art-hero-warm',
  'docs/CONTRIBUTIONS.html': ' rt-art-hero-wide rt-art-hero-cool'
};

function cover(doc) {
  var variant = COVER_VARIANT[doc.html] || '';
  return [
    '    <figure class="rt-art-hero rt-doc-cover rounded-2xl' + variant + '">',
    '      <img src="' + COVER + '-768.webp" srcset="' + COVER + '-768.webp 768w, ' + COVER + '-1280.webp 1280w" sizes="(max-width: 840px) 100vw, 800px" width="1280" height="720" alt="' + esc(COVER_ALT) + '" loading="eager" fetchpriority="high" decoding="async">',
    '    </figure>'
  ].join('\n');
}

function esc(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

/* ---------- links ---------- */
function prettyFor(repoPath) {
  for (var i = 0; i < DOCS.length; i++) if (DOCS[i].md === repoPath) return DOCS[i].html;
  return null;
}

function safeHref(href, fromMd) {
  href = String(href).trim();
  if (/^#/.test(href)) return href;
  if (/^mailto:/i.test(href)) return href;
  if (/^https?:\/\//i.test(href)) return href;
  if (/^[a-z][a-z0-9+.-]*:/i.test(href) || /^\/\//.test(href)) return null; /* javascript:, data:, etc. */
  if (!fromMd) return href;
  /* Relative repo path: a styled sibling gets its page; anything else opens on GitHub. */
  var hash = '';
  var h = href.indexOf('#');
  if (h !== -1) { hash = href.slice(h); href = href.slice(0, h); }
  var base = href.charAt(0) === '/' ? '' : path.posix.dirname(fromMd);
  var target = path.posix.normalize(path.posix.join(base, href.replace(/^\//, '')));
  if (target.indexOf('..') === 0) return null;
  var pretty = prettyFor(target);
  if (pretty) return '/' + pretty + hash;
  return REPO_BLOB + target.split('/').map(encodeURIComponent).join('/') + hash;
}

/* ---------- inline ---------- */
function inline(src, ctx, shared, inLink) {
  var slots = shared || [];
  function slot(html) { slots.push(html); return '\u0000' + (slots.length - 1) + '\u0000'; }
  var s = String(src);
  /* backslash escapes */
  s = s.replace(/\\([\\`*_{}\[\]()#+\-.!|>])/g, function (_, c) { return slot(esc(c)); });
  /* code spans */
  s = s.replace(/(`+)([\s\S]*?[^`])\1(?!`)/g, function (_, ticks, code) {
    return slot('<code>' + esc(code.replace(/^ (.*) $/, '$1')) + '</code>');
  });
  /* links [text](href) */
  s = s.replace(/\[((?:[^\[\]]|\[[^\[\]]*\])+)\]\(\s*<?([^()\s<>]+(?:\([^()\s]*\)[^()\s<>]*)*)>?(?:\s+"[^"]*")?\s*\)/g, function (_, text, href) {
    var inner = inline(text, ctx, slots, true);
    var safe = safeHref(href, ctx.md);
    if (safe === null) return slot(inner);
    return slot('<a href="' + esc(safe) + '">' + inner + '</a>');
  });
  /* autolinks <https://…> */
  s = s.replace(/<(https?:\/\/[^\s<>]+|mailto:[^\s<>]+)>/gi, function (_, u) {
    return slot('<a href="' + esc(u) + '">' + esc(u) + '</a>');
  });
  /* bare URLs (not inside link text: no nested anchors) */
  if (!inLink) s = s.replace(/https?:\/\/[^\s<>"'`\u0000]+/g, function (u) {
    var trail = '';
    var m = u.match(/[.,;:!?)\]*_]+$/);
    if (m) {
      /* keep a closing paren that balances one inside the URL */
      trail = m[0];
      u = u.slice(0, u.length - trail.length);
      if (trail.charAt(0) === ')' && u.indexOf('(') !== -1) { u += ')'; trail = trail.slice(1); }
    }
    return slot('<a href="' + esc(u) + '">' + esc(u) + '</a>') + trail;
  });
  s = esc(s);
  /* emphasis */
  s = s.replace(/\*\*(?=\S)([\s\S]*?\S)\*\*/g, '<strong>$1</strong>');
  s = s.replace(/(^|[^\w])__(?=\S)([\s\S]*?\S)__(?=[^\w]|$)/g, '$1<strong>$2</strong>');
  s = s.replace(/(^|[^*\w])\*(?=[^\s*])([^*]*?[^\s*])\*(?!\*)/g, '$1<em>$2</em>');
  s = s.replace(/(^|[^\w])_(?=\S)([^_]*?\S)_(?=[^\w]|$)/g, '$1<em>$2</em>');
  /* hard line break: two trailing spaces */
  s = s.replace(/ {2,}\n/g, '<br>\n');
  if (shared) return s;
  var guard = 0;
  while (/\u0000\d+\u0000/.test(s) && guard++ < 20) {
    s = s.replace(/\u0000(\d+)\u0000/g, function (_, n) { return slots[+n]; });
  }
  return s;
}

function plain(src) {
  return inline(src, { md: null }).replace(/<[^>]+>/g, '')
    .replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&quot;/g, '"').replace(/&#39;/g, "'").replace(/&amp;/g, '&');
}

function slugify(text, used) {
  var base = plain(text).toLowerCase().trim()
    .replace(/[^\p{L}\p{N}\s-]/gu, '')
    .replace(/\s/g, '-') || 'section';
  var slug = base;
  var n = 1;
  while (used[slug]) slug = base + '-' + n++;
  used[slug] = 1;
  return slug;
}

/* ---------- blocks ---------- */
var RE_FENCE = /^ {0,3}(`{3,}|~{3,})\s*([^`\s]*)[^`]*$/;
var RE_HEAD = /^ {0,3}(#{1,6})\s+(.*?)\s*#*\s*$/;
var RE_HR = /^ {0,3}([-*_])(\s*\1){2,}\s*$/;
var RE_LIST = /^( {0,3})([-*+]|\d{1,9}[.)])\s+(.*)$/;
var RE_QUOTE = /^ {0,3}>\s?(.*)$/;
var RE_TSEP = /^\s*\|?\s*:?-{1,}:?\s*(\|\s*:?-{1,}:?\s*)*\|?\s*$/;

function splitRow(line) {
  var t = line.trim();
  if (t.charAt(0) === '|') t = t.slice(1);
  if (t.charAt(t.length - 1) === '|' && t.charAt(t.length - 2) !== '\\') t = t.slice(0, -1);
  var cells = [];
  var cur = '';
  var inCode = false;
  for (var i = 0; i < t.length; i++) {
    var c = t.charAt(i);
    if (c === '\\' && t.charAt(i + 1) === '|') { cur += '|'; i++; continue; }
    if (c === '`') inCode = !inCode;
    if (c === '|' && !inCode) { cells.push(cur.trim()); cur = ''; continue; }
    cur += c;
  }
  cells.push(cur.trim());
  return cells;
}

function isBlockStart(line, next) {
  return RE_FENCE.test(line) || RE_HEAD.test(line) || RE_HR.test(line) || RE_LIST.test(line) ||
    RE_QUOTE.test(line) || (/\|/.test(line) && next !== undefined && RE_TSEP.test(next) && /-/.test(next));
}

function blocks(lines, ctx) {
  var out = [];
  var i = 0;
  while (i < lines.length) {
    var line = lines[i];
    var m;
    if (!line.trim()) { i++; continue; }

    if ((m = line.match(RE_FENCE))) {
      var fence = m[1];
      var lang = m[2];
      var body = [];
      i++;
      while (i < lines.length && !(lines[i].trim().indexOf(fence) === 0 && lines[i].trim().replace(/[`~]/g, '') === '')) {
        body.push(lines[i]);
        i++;
      }
      i++;
      out.push('<pre class="rt-doc-code"' + (lang ? ' data-lang="' + esc(lang) + '"' : '') + '><code>' +
        esc(body.join('\n')) + '</code></pre>');
      continue;
    }

    if ((m = line.match(RE_HEAD))) {
      var level = m[1].length;
      var text = m[2];
      if (level === 1 && !ctx.title) {
        ctx.title = plain(text);
        out.push('<h1 id="' + esc(slugify(text, ctx.ids)) + '">' + inline(text, ctx) + '</h1>');
      } else {
        var id = slugify(text, ctx.ids);
        if (level === 2) ctx.toc.push({ id: id, html: inline(text, ctx) });
        out.push('<h' + level + ' id="' + esc(id) + '">' + inline(text, ctx) +
          ' <a class="rt-doc-anchor" href="#' + esc(id) + '" aria-hidden="true" tabindex="-1">#</a></h' + level + '>');
      }
      i++;
      continue;
    }

    if (RE_HR.test(line)) { out.push('<hr>'); i++; continue; }

    if (/\|/.test(line) && i + 1 < lines.length && RE_TSEP.test(lines[i + 1]) && /-/.test(lines[i + 1])) {
      var head = splitRow(line);
      var aligns = splitRow(lines[i + 1]).map(function (c) {
        var l = c.charAt(0) === ':';
        var r = c.charAt(c.length - 1) === ':';
        return l && r ? 'center' : r ? 'right' : l ? 'left' : '';
      });
      i += 2;
      var rows = [];
      while (i < lines.length && lines[i].trim() && /\|/.test(lines[i])) { rows.push(splitRow(lines[i])); i++; }
      function cell(tag, c, k) {
        var a = aligns[k] ? ' style="text-align:' + aligns[k] + '"' : '';
        return '<' + tag + a + '>' + inline(c, ctx) + '</' + tag + '>';
      }
      var t = '<div class="rt-doc-table" role="region" aria-label="' + esc('Table ' + (++ctx.tables)) + '" tabindex="0"><table>\n<thead><tr>' +
        head.map(function (c, k) { return cell('th', c, k); }).join('') + '</tr></thead>\n<tbody>\n';
      rows.forEach(function (r) {
        var cells = [];
        for (var k = 0; k < head.length; k++) cells.push(cell('td', r[k] === undefined ? '' : r[k], k));
        t += '<tr>' + cells.join('') + '</tr>\n';
      });
      out.push(t + '</tbody>\n</table></div>');
      continue;
    }

    if (RE_QUOTE.test(line)) {
      var q = [];
      while (i < lines.length && lines[i].trim() && (m = lines[i].match(RE_QUOTE))) { q.push(m[1]); i++; }
      out.push('<blockquote>\n' + blocks(q, ctx) + '\n</blockquote>');
      continue;
    }

    if ((m = line.match(RE_LIST))) {
      var ordered = /\d/.test(m[2]);
      var start = ordered ? parseInt(m[2], 10) : 1;
      var items = [];
      while (i < lines.length) {
        var lm = lines[i].match(RE_LIST);
        if (lm && /\d/.test(lm[2]) === ordered) { items.push([lm[3]]); i++; continue; }
        if (lm) break;
        if (!lines[i].trim()) {
          /* a blank line ends the list unless an indented continuation follows */
          if (i + 1 < lines.length && /^\s{2,}\S/.test(lines[i + 1]) && !RE_LIST.test(lines[i + 1])) { i++; continue; }
          break;
        }
        if (isBlockStart(lines[i], lines[i + 1]) && !/^\s{2,}/.test(lines[i])) break;
        items[items.length - 1].push(lines[i].replace(/^\s+/, ''));
        i++;
      }
      var tag = ordered ? 'ol' : 'ul';
      out.push('<' + tag + (ordered && start !== 1 ? ' start="' + start + '"' : '') + '>\n' +
        items.map(function (it) { return '<li>' + inline(it.join('\n'), ctx) + '</li>'; }).join('\n') +
        '\n</' + tag + '>');
      continue;
    }

    var para = [];
    while (i < lines.length && lines[i].trim() && !(para.length && isBlockStart(lines[i], lines[i + 1]))) {
      para.push(lines[i]);
      i++;
    }
    out.push('<p>' + inline(para.join('\n').replace(/^\s+/, ''), ctx) + '</p>');
  }
  return out.join('\n');
}

function renderMarkdown(md, mdPath) {
  var ctx = { md: mdPath || null, title: '', ids: Object.create(null), toc: [], tables: 0 };
  var html = blocks(String(md).replace(/\r\n?/g, '\n').split('\n'), ctx);
  return { html: html, title: ctx.title, toc: ctx.toc };
}

/* ---------- page ---------- */
function page(doc, r) {
  var title = r.title || path.basename(doc.md);
  var url = SITE + doc.html;
  var raw = '/' + doc.md;
  var desc = 'Readable copy of ' + doc.md + ' from the Ra-Thor monorepo. The raw markdown stays at ' + raw + '.';
  var others = DOCS.filter(function (d) { return d !== doc; }).map(function (d) {
    return '<li><a href="/' + esc(d.html) + '">' + esc(titleOf(d)) + '</a></li>';
  }).join('\n          ');
  var toc = r.toc.length >= 3
    ? '      <nav class="rt-doc-toc" aria-label="On this page">\n        <p class="rt-doc-toc-label">On this page</p>\n        <ul>\n' +
      r.toc.map(function (t) { return '          <li><a href="#' + esc(t.id) + '">' + t.html.replace(/<a [^>]*>|<\/a>/g, '') + '</a></li>'; }).join('\n') +
      '\n        </ul>\n      </nav>\n'
    : '';
  return [
    '<!DOCTYPE html>',
    '<!-- Generated by scripts/docs-pretty/build.js from ' + doc.md + '. Do not edit by hand: edit the .md, then run the script. -->',
    '<html lang="en" dir="ltr" data-theme="dark">',
    '<head>',
    '  <meta charset="UTF-8">',
    '  <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=yes">',
    '  <title>' + esc(title) + ' — Ra-Thor</title>',
    '  <meta name="description" content="' + esc(desc) + '">',
    '  <meta name="theme-color" content="#050505">',
    '  <meta name="color-scheme" content="dark light">',
    '  <meta property="og:type" content="article">',
    '  <meta property="og:url" content="' + esc(url) + '">',
    '  <meta property="og:title" content="' + esc(title) + ' — Ra-Thor">',
    '  <meta property="og:description" content="' + esc(desc) + '">',
    '  <meta property="og:image" content="https://rathor.ai/icons/ra-thor-og-1200x630.png">',
    '  <meta name="twitter:card" content="summary_large_image">',
    '  <meta name="twitter:image" content="https://rathor.ai/icons/ra-thor-og-1200x630.png">',
    '  <link rel="canonical" href="' + esc(url) + '">',
    '  <link rel="alternate" type="text/markdown" href="' + esc(raw) + '">',
    '  <link rel="manifest" href="/manifest.json">',
    '  <link rel="icon" href="/icons/ra-thor-icon-192.png" type="image/png">',
    '  <link rel="apple-touch-icon" href="/icons/ra-thor-icon-192.png">',
    '  <link rel="stylesheet" href="/css/rathor-theme.css">',
    '  <link rel="stylesheet" href="/css/rathor-docs.css">',
    '  <script>',
    '    (function () {',
    "      try { document.documentElement.setAttribute('data-theme', localStorage.getItem('rathor-theme') || 'dark');",
    "      } catch (e) { document.documentElement.setAttribute('data-theme', 'dark'); }",
    '    })();',
    '  </script>',
    '</head>',
    '<body class="bg-black text-white overflow-x-hidden rt-doc-body">',
    '  <main class="rt-doc-main" id="rt-family-main" dir="ltr" lang="en" data-rt-prose>',
    '    <div class="rt-doc-bar">',
    '      <a class="rt-doc-home" href="/">← Home</a>',
    (doc.source === 'github'
      ? '      <a class="rt-doc-raw" href="' + esc(REPO_BLOB + doc.md) + '">Read the source on GitHub</a>'
      : '      <a class="rt-doc-raw" href="' + esc(raw) + '" type="text/markdown">View raw markdown</a>'),
    '    </div>',
    cover(doc),
    '    <div class="rt-doc-layout">',
    toc + '      <article class="rt-doc rt-prose" data-doc-source="' + esc(doc.md) + '">',
    r.html,
    '      </article>',
    '    </div>',
    '    <nav class="rt-doc-more" aria-label="More documents">',
    '      <p class="rt-doc-toc-label">More documents</p>',
    '      <ul>',
    '          ' + others,
    '      </ul>',
    '    </nav>',
    '    <p class="rt-doc-source">Source: <a href="' + esc(raw) + '">' + esc(doc.md) + '</a> · <a href="' + esc(REPO_BLOB + doc.md) + '">GitHub</a> · <a href="/">rathor.ai</a></p>',
    '  </main>',
    '  <script src="/i18n/en.js?v=20260924a"></script>',
    '  <script src="/js/i18n-chrome.js?v=20260924a"></script>',
    '  <script src="/js/rathor-theme.js"></script>',
    '  <script src="/js/pwa-install.js"></script>',
    '  <script src="/js/family-nav-2026-08-22.js"></script>',
    '  <script src="/js/site-lock-2026-08-22.js"></script>',
    '</body>',
    '</html>',
    ''
  ].join('\n');
}

var titleCache = {};
function titleOf(doc) {
  if (!titleCache[doc.md]) {
    var md = fs.readFileSync(path.join(ROOT, doc.md), 'utf8');
    titleCache[doc.md] = renderMarkdown(md, doc.md).title || path.basename(doc.md);
  }
  return titleCache[doc.md];
}

function build(doc) {
  var md = fs.readFileSync(path.join(ROOT, doc.md), 'utf8');
  return page(doc, renderMarkdown(md, doc.md));
}

function main(argv) {
  var check = argv.indexOf('--check') !== -1;
  var drift = [];
  DOCS.forEach(function (doc) {
    var html = build(doc);
    var out = path.join(ROOT, doc.html);
    if (check) {
      var cur = fs.existsSync(out) ? fs.readFileSync(out, 'utf8') : null;
      if (cur !== html) drift.push(doc.html);
    } else {
      fs.writeFileSync(out, html);
      console.log('wrote ' + doc.html);
    }
  });
  if (check) {
    if (drift.length) {
      console.error('docs-pretty: out of date: ' + drift.join(', ') + ' — run node scripts/docs-pretty/build.js');
      process.exit(1);
    }
    console.log('docs-pretty: ' + DOCS.length + ' pages match their .md');
  }
}

module.exports = { DOCS: DOCS, COVER_VARIANT: COVER_VARIANT, renderMarkdown: renderMarkdown, build: build, inline: inline, safeHref: safeHref };
if (require.main === module) main(process.argv.slice(2));
