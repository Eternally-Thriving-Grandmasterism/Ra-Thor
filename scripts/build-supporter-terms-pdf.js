#!/usr/bin/env node
/* GOLIVE-PREP-1: build the frozen Supporter Pass terms PDF (terms/supporter-pass-v1.0.pdf) from supporter-terms.html.
 *
 *   node scripts/build-supporter-terms-pdf.js --date 2026-12-01            # writes terms/supporter-pass-v1.0.pdf
 *   node scripts/build-supporter-terms-pdf.js --out /tmp/x.pdf --force      # scratch build (no date: fixed 1970-01-01 metadata)
 *
 * What goes in: the terms page's own text, verbatim: heading, version line and sections 1-12. Left out: the page's
 * "not offered yet" notice paragraph, comments, site chrome and the site footer line. Visible text is unchanged;
 * site-relative hrefs are rewritten to https://rathor.ai/... so the PDF has no file:// links.
 * Deterministic: PDF /CreationDate and /ModDate are fixed from --date (same byte length, so offsets stay valid).
 * The bytes still depend on the Chrome build (its version string is in /Creator) and on the installed serif fonts, so a
 * rebuild on another machine can differ byte-for-byte; the text never does. Build ONCE at go-live, commit, never edit:
 * a new version gets a new file name (-v1.1.pdf, ...).
 * Needs Chrome or Chromium (env CHROME, or google-chrome / chromium on PATH). Set CHROME_NO_SANDBOX=1 when running as root. */
var fs = require('fs');
var os = require('os');
var path = require('path');
var cp = require('child_process');
var crypto = require('crypto');

var root = path.join(__dirname, '..');
var DEFAULT_OUT = 'terms/supporter-pass-v1.0.pdf';

function absolutiseHref(href) {
  if (/^(https?:|mailto:|tel:)/i.test(href)) return href;
  if (href.indexOf('//') === 0) return 'https:' + href;
  if (href.charAt(0) === '/') return 'https://rathor.ai' + href;
  return 'https://rathor.ai/' + href;
}

function extractTermsHtml(page) {
  var m = page.match(/<main[^>]*>([\s\S]*?)<\/main>/);
  if (!m) throw new Error('supporter-terms.html has no <main>');
  var body = m[1]
    .replace(/<!--[\s\S]*?-->/g, '')
    // the page's notice paragraph (whatever its wording) is not part of the terms
    .replace(/<p class="text-sm text-white\/70 mt-4">[\s\S]*?<\/p>/, '')
    // the site footer line under the sections
    .replace(/<p class="text-white\/50 text-sm text-center">[\s\S]*?<\/p>/, '')
    .replace(/\sclass="[^"]*"/g, '')
    .replace(/\s(?:role|dir|lang|id)="[^"]*"/g, '')
    // site-relative hrefs must be absolute, or Chrome print-to-pdf writes file:// annotations
    .replace(/href="([^"]*)"/g, function (_, href) { return 'href="' + absolutiseHref(href) + '"'; });
  return body;
}

function pageHtml(body) {
  return '<!doctype html><html lang="en"><head><meta charset="utf-8"><base href="https://rathor.ai/"><title>Ra-Thor Supporter Pass: Terms</title><style>' +
    '@page { size: Letter; margin: 18mm; }' +
    'body { font: 10.5pt/1.45 "Times New Roman", Times, "Liberation Serif", serif; color: #000; background: #fff; }' +
    'h1 { font-size: 18pt; margin: 0 0 4pt; } h2 { font-size: 11.5pt; margin: 12pt 0 3pt; }' +
    'p { margin: 0 0 6pt; } ul { margin: 0 0 6pt 16pt; padding: 0; } li { margin: 0 0 4pt; }' +
    'table { border-collapse: collapse; margin: 4pt 0 8pt; } th, td { border: 1px solid #000; padding: 2pt 8pt; text-align: left; }' +
    'a { color: #000; text-decoration: underline; } section, li, tr { break-inside: avoid; }' +
    '</style></head><body>' + body + '</body></html>';
}

function findChrome() {
  var cands = [process.env.CHROME, 'google-chrome', 'google-chrome-stable', 'chromium', 'chromium-browser'].filter(Boolean);
  for (var i = 0; i < cands.length; i++) {
    var r = cp.spawnSync(cands[i], ['--version'], { encoding: 'utf8' });
    if (r.status === 0) return cands[i];
  }
  throw new Error('Chrome or Chromium not found (set CHROME=/path/to/chrome)');
}

function normaliseDates(buf, isoDate) {
  var d = (isoDate || '1970-01-01').replace(/-/g, '');
  if (!/^\d{8}$/.test(d)) throw new Error('bad date ' + isoDate);
  var fixed = "D:" + d + "000000+00'00'";
  var s = buf.toString('latin1');
  var n = 0;
  s = s.replace(/\/(CreationDate|ModDate) \(D:\d{14}[^)]*\)/g, function (m, k) {
    var out = '/' + k + ' (' + fixed + ')';
    if (out.length !== m.length) throw new Error('date field length differs: ' + m);
    n++;
    return out;
  });
  if (n < 1) throw new Error('no PDF date fields found');
  return Buffer.from(s, 'latin1');
}

function buildPdf(opts) {
  opts = opts || {};
  var rootDir = opts.root || root;
  var out = path.resolve(rootDir, opts.out || DEFAULT_OUT);
  if (fs.existsSync(out) && !opts.force) throw new Error(out + ' exists; the frozen PDF is never overwritten (use a new version name, or --force for a scratch build)');
  var page = fs.readFileSync(path.join(rootDir, 'supporter-terms.html'), 'utf8');
  var html = pageHtml(extractTermsHtml(page));
  var tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'sp-terms-'));
  var htmlPath = path.join(tmp, 'terms.html');
  var rawPdf = path.join(tmp, 'terms.pdf');
  fs.writeFileSync(htmlPath, html);
  var args = ['--headless=new', '--disable-gpu', '--no-pdf-header-footer', '--print-to-pdf=' + rawPdf, 'file://' + htmlPath];
  if (process.env.CHROME_NO_SANDBOX === '1' || (process.getuid && process.getuid() === 0)) args.unshift('--no-sandbox');
  var r = cp.spawnSync(findChrome(), args, { encoding: 'utf8', timeout: 120000 });
  if (!fs.existsSync(rawPdf)) throw new Error('Chrome did not write a PDF: ' + (r.stderr || '').slice(0, 300));
  var buf = normaliseDates(fs.readFileSync(rawPdf), opts.date);
  fs.mkdirSync(path.dirname(out), { recursive: true });
  fs.writeFileSync(out, buf);
  fs.rmSync(tmp, { recursive: true, force: true });
  return { out: out, sha256: crypto.createHash('sha256').update(buf).digest('hex'), bytes: buf.length };
}

module.exports = { buildPdf: buildPdf, extractTermsHtml: extractTermsHtml };

if (require.main === module) {
  var argv = process.argv.slice(2);
  var o = {};
  for (var i = 0; i < argv.length; i++) {
    if (argv[i] === '--date') o.date = argv[++i];
    else if (argv[i] === '--out') o.out = argv[++i];
    else if (argv[i] === '--force') o.force = true;
    else { console.error('unknown arg ' + argv[i]); process.exit(2); }
  }
  try {
    var res = buildPdf(o);
    console.log('wrote ' + path.relative(process.cwd(), res.out) + ' (' + res.bytes + ' bytes) sha256 ' + res.sha256);
  } catch (e) { console.error('ERROR: ' + e.message); process.exit(1); }
}
