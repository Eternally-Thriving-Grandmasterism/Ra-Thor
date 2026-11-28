#!/usr/bin/env node
/* GOLIVE-PREP-1: the go-live swap. Small, mechanical, one push.
 *
 * Dry run (default, writes nothing): checks every swap spot and every "old" line is present exactly as expected, lists what
 * is still missing (values, button labels, go-live copy) and says READY or NOT READY.
 *
 *   node scripts/golive-swap.js
 *   node scripts/golive-swap.js --write --date 2026-12-01 --discord-invite <https://discord.gg/...> \
 *        --paylink-100 <url> --paylink-400 <url> --paylink-sp-1m <url> --paylink-sp-3m <url> --paylink-sp-12m <url>
 *
 * Values may also come from env: GOLIVE_DATE, GOLIVE_DISCORD_INVITE, GOLIVE_PAYLINK_100, GOLIVE_PAYLINK_400,
 * GOLIVE_PAYLINK_SP_1M, GOLIVE_PAYLINK_SP_3M, GOLIVE_PAYLINK_SP_12M. Only the Stripe account owner supplies the five links;
 * this script never calls Stripe and never invents a value, a label or a line of copy. --write refuses to run while any
 * "new" in scripts/golive-swap.config.json (or any button label) is null.
 * --write on a clean tree does, in order: five hrefs (inert span becomes <a>), Discord invite, date (terms pages and the
 * privacy "Last Updated" line in privacy.html and the 23 packs), the configured copy flips, the frozen PDF
 * (scripts/build-supporter-terms-pdf.js), then the test pins (lock test: positive host pins and flipped lines;
 * art-uplift-3 and footer-i18n-1: sha256 re-pin). Review `git diff`, run `node tests/run-node-tests.js`, then push.
 * --root DIR and --config FILE exist so the swap can be rehearsed on a scratch copy with fake values. */
var fs = require('fs');
var path = require('path');
var crypto = require('crypto');

var HOST = ['buy', 'stripe', 'com'].join('.'); // built by parts: the repo lock test forbids the literal outside tests/

function parseArgs(argv) {
  var o = { write: false, vals: {} };
  var map = { '--paylink-100': 'PAYLINK_100', '--paylink-400': 'PAYLINK_400', '--paylink-sp-1m': 'PAYLINK_SP_1M', '--paylink-sp-3m': 'PAYLINK_SP_3M',
    '--paylink-sp-12m': 'PAYLINK_SP_12M', '--discord-invite': 'DISCORD_INVITE', '--date': 'GOLIVE_DATE' };
  for (var i = 0; i < argv.length; i++) {
    var a = argv[i];
    if (a === '--write') o.write = true;
    else if (a === '--root') o.root = argv[++i];
    else if (a === '--config') o.config = argv[++i];
    else if (map[a]) o.vals[map[a]] = argv[++i];
    else throw new Error('unknown argument ' + a);
  }
  Object.keys(map).forEach(function (k) {
    var id = map[k];
    var ev = process.env['GOLIVE_' + id.replace(/^GOLIVE_/, '')];
    if (!o.vals[id] && ev) o.vals[id] = ev;
  });
  return o;
}

function validPayLink(u) {
  var p;
  try { p = new URL(u); } catch (e) { return false; }
  return p.protocol === 'https:' && p.hostname === HOST && !p.username && !p.password && !p.port && !p.search && !p.hash &&
    /^\/[A-Za-z0-9]{6,64}$/.test(p.pathname) && !/test_/i.test(u);
}
function validInvite(u) {
  var p;
  try { p = new URL(u); } catch (e) { return false; }
  return p.protocol === 'https:' && !p.username && !p.password && !p.search && !p.hash && !p.port &&
    ((p.hostname === 'discord.gg' && /^\/[A-Za-z0-9-]{2,32}$/.test(p.pathname)) ||
     (p.hostname === 'discord.com' && /^\/invite\/[A-Za-z0-9-]{2,32}$/.test(p.pathname)));
}
function validDate(d) {
  return /^\d{4}-\d{2}-\d{2}$/.test(d) && !isNaN(Date.parse(d + 'T00:00:00Z')) && new Date(d + 'T00:00:00Z').toISOString().slice(0, 10) === d;
}
function jsq(s) { return String(s).replace(/\\/g, '\\\\').replace(/'/g, "\\'"); }
function countOf(hay, needle) { return needle ? hay.split(needle).length - 1 : 0; }
function sha(buf) { return crypto.createHash('sha256').update(buf).digest('hex'); }

function main() {
  var o = parseArgs(process.argv.slice(2));
  var root = path.resolve(o.root || path.join(__dirname, '..'));
  var cfg = JSON.parse(fs.readFileSync(o.config || path.join(__dirname, 'golive-swap.config.json'), 'utf8'));
  var files = {};
  function rd(rel) { if (!(rel in files)) files[rel] = fs.readFileSync(path.join(root, rel), 'utf8'); return files[rel]; }
  function wr(rel, s) { files[rel] = s; }
  var problems = [];
  var missing = [];

  // 1. every swap spot present, once, with its marker
  cfg.swapSpots.forEach(function (sp) {
    (sp.files || [sp.file]).forEach(function (f) {
      var t = rd(f);
      var live = countOf(t.replace(/<!--[\s\S]*?-->/g, ''), sp.token); // the marker comment names the token too
      if (live !== 1) problems.push(sp.id + ': expected token ' + sp.token + ' once outside comments in ' + f + ', found ' + live);
      if (countOf(t, 'GOLIVE-SWAP ' + sp.id + ':') !== 1) problems.push(sp.id + ': marker comment "GOLIVE-SWAP ' + sp.id + ':" missing in ' + f);
    });
  });
  // 2. values
  var V = o.vals;
  cfg.swapSpots.forEach(function (sp) {
    var v = V[sp.id];
    if (!v) { missing.push('value ' + sp.id); return; }
    var ok = sp.kind === 'paylink' ? validPayLink(v) : sp.kind === 'discord' ? validInvite(v) : validDate(v);
    if (!ok) problems.push(sp.id + ': value rejected (' + (sp.kind === 'paylink' ? 'must be https on the Stripe Payment Link host, plain /id path, no userinfo/query/port, not a test link' : sp.kind === 'discord' ? 'must be https://discord.gg/<code> or https://discord.com/invite/<code>' : 'must be YYYY-MM-DD') + ')');
  });
  var payIds = cfg.swapSpots.filter(function (s) { return s.kind === 'paylink'; }).map(function (s) { return s.id; });
  var seen = {};
  payIds.forEach(function (id) { if (V[id]) { if (seen[V[id]]) problems.push(id + ' repeats another link'); seen[V[id]] = 1; } });
  // 3. labels and copy
  payIds.forEach(function (id) { if (cfg.labels[id] === null || cfg.labels[id] === undefined) missing.push('button label ' + id); });
  var edits = [];
  cfg.flips.forEach(function (fl) {
    fl.edits.forEach(function (e) {
      var targets = e.glob ? fs.readdirSync(path.join(root, path.dirname(e.glob))).filter(function (n) { return /\.js$/.test(n); }).map(function (n) { return path.dirname(e.glob) + '/' + n; }) : [e.file];
      var hits = 0;
      targets.forEach(function (f) {
        var c = countOf(rd(f), e.old);
        if (e.glob) { hits += c ? 1 : 0; if (c && c !== 1) problems.push(fl.id + '/' + e.id + ': ' + f + ' has ' + c + ' hits, expected 1'); }
        else if (c !== (e.count || 1)) problems.push(fl.id + '/' + e.id + ': ' + f + ' has ' + c + ' hits of the old line, expected ' + (e.count || 1));
      });
      if (e.glob && hits === 0) problems.push(fl.id + '/' + e.id + ': old line not found in any pack');
      if (e.glob && !e.optional && hits !== targets.length) problems.push(fl.id + '/' + e.id + ': found in ' + hits + ' of ' + targets.length + ' packs');
      if (e.new === null || e.new === undefined) missing.push('go-live copy for ' + fl.id + '/' + e.id + ' (' + fl.source + ')');
      edits.push({ flip: fl, e: e, targets: targets });
    });
    fl.pins.forEach(function (p) { if (countOf(rd(p.file), p.old) !== 1) problems.push(fl.id + ': test pin line not found exactly once in ' + p.file); });
  });
  // fixed (non-copy) lock-test lines
  var LOCK = 'tests/licence-self-serve-lock.test.js';
  var lock = rd(LOCK);
  var lockOld = {
    strip: "  var text = read(f);\n  if (f.indexOf('workers/') !== 0) {",
    siteCount: 'var siteCount = 0;\n',
    l119: "assert(licence.indexOf('aria-disabled=\"true\"') !== -1, 'licence.html checkout control must be disabled');\n",
    l127: "assert(tos.indexOf('last updated [GO-LIVE MERGE DATE]') !== -1, 'terms date stays a placeholder until go-live');\n",
    l152: "assert(/\\[TEST_PAYMENT_LINK_100\\]/.test(licence) && /\\[TEST_PAYMENT_LINK_400\\]/.test(licence), 'checkout placeholders stay placeholders until go-live');\n",
    sp: /\['\[TEST_PAYMENT_LINK_SP_1M\]'[\s\S]*?'three inert checkout spans, no href'\);\n/,
    l201: "assert(!/\\bhref=\"(?!mailto:|tel:|\\/supporter-terms\\.html|\\/privacy\\.html|https:\\/\\/rathor\\.ai)/.test(sp), 'supporter section links only to mailto, tel, the terms and privacy pages');\n",
    l210: "assert(sterms.indexOf('Version 1.0 · Effective [GO-LIVE MERGE DATE]') !== -1, 'supporter terms date stays a placeholder until go-live');\n",
    l211: "assert(sterms.indexOf('[DISCORD INVITE LINK]') !== -1, 'Discord invite stays a placeholder until Sherif provides the link');\n",
    l218: "assert(!/discord\\.gg|discord\\.com\\/invite|discordapp\\.com\\/invite/i.test(sp + sterms + supporters + priv), 'no real Discord invite link in the supporter pages; the invite stays a placeholder');\n"
  };
  Object.keys(lockOld).forEach(function (k) {
    var c = lockOld[k] instanceof RegExp ? (lock.match(new RegExp(lockOld[k].source, 'g')) || []).length : countOf(lock, lockOld[k]);
    if (c !== 1) problems.push('lock test line "' + k + '" not found exactly once (' + c + ')');
  });

  // report
  console.log('golive-swap: ' + cfg.swapSpots.length + ' swap spots, ' + cfg.flips.length + ' copy flips checked in ' + root);
  problems.forEach(function (p) { console.log('PROBLEM: ' + p); });
  missing.forEach(function (m) { console.log('MISSING: ' + m); });
  if (problems.length) { console.log('NOT READY: fix the problems above (the tree no longer matches the config).'); process.exit(2); }
  if (missing.length) {
    console.log('NOT READY: ' + missing.length + ' missing item(s).' + (o.write ? ' --write refused; nothing was changed.' : ''));
    process.exit(o.write ? 3 : 0);
  }
  if (!o.write) { console.log('READY: rerun with --write to apply.'); return; }

  // ---- apply (all in memory first) ----
  var date = V.GOLIVE_DATE;
  var spanRe = function (tok) { return new RegExp('( *)<!-- GOLIVE-SWAP [A-Z0-9_]+: [^>]*?Token ' + tok.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + '\\.[^>]*?-->\\n( *)<span class="rt-btn-ghost" role="link" aria-disabled="true" data-checkout-test-link="' + tok.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + '"[^>]*>[^<]*</span>'); };
  var links = {};
  cfg.swapSpots.forEach(function (sp) {
    var v = V[sp.id];
    if (sp.kind === 'paylink') {
      var t = rd(sp.file);
      var re = spanRe(sp.token);
      if (!re.test(t)) throw new Error(sp.id + ': inert span not in the expected shape');
      t = t.replace(re, function (m, ind1, ind2) {
        return ind2 + '<a class="rt-btn-ghost" href="' + v + '" rel="noopener" target="_blank" data-golive-link="' + sp.id + '">' + cfg.labels[sp.id] + '</a>';
      });
      wr(sp.file, t);
      (links[sp.file] = links[sp.file] || {})[sp.id] = v;
    } else if (sp.kind === 'discord') {
      var d = rd(sp.file);
      d = d.replace(/ *<!-- GOLIVE-SWAP DISCORD_INVITE:[\s\S]*?-->\n/, '');
      d = d.replace(sp.token, '<a href="' + v + '" rel="noopener" target="_blank">' + v + '</a>');
      wr(sp.file, d);
    } else {
      sp.files.forEach(function (f) {
        var x = rd(f).replace(/ *<!-- GOLIVE-SWAP GOLIVE_DATE:[\s\S]*?-->\n/, '').replace(sp.token, date);
        wr(f, x);
      });
    }
  });
  // privacy date
  var pd = cfg.privacyDate;
  var ph = rd(pd.html);
  var phLines = ph.split('\n').map(function (ln) { return ln.indexOf('"' + pd.packKey + '"') !== -1 && ln.indexOf(pd.old) !== -1 ? ln.replace(pd.old, date) : ln; });
  var phn = phLines.join('\n');
  if (countOf(phn, pd.old) === countOf(ph, pd.old)) {
    // fall back: the visible line in privacy.html is the data-i18n line carrying the old date
    var phLines2 = ph.split('\n').map(function (ln) { return /data-i18n="privacyUpdated"/.test(ln) && ln.indexOf(pd.old) !== -1 ? ln.replace(pd.old, date) : ln; });
    phn = phLines2.join('\n');
  }
  if (phn === ph) throw new Error('privacy.html Last Updated line not found');
  wr(pd.html, phn);
  var packNames = fs.readdirSync(path.join(root, 'i18n')).filter(function (n) { return /\.js$/.test(n); });
  var packHits = 0;
  packNames.forEach(function (n) {
    var rel = 'i18n/' + n;
    var s = rd(rel);
    var out = s.split('\n').map(function (ln) { return /\bprivacyUpdated\b/.test(ln) && ln.indexOf(pd.old) !== -1 ? ln.replace(pd.old, date) : ln; }).join('\n');
    if (out !== s) packHits++;
    wr(rel, out);
  });
  if (packHits !== packNames.length) throw new Error('privacyUpdated date found in ' + packHits + ' of ' + packNames.length + ' packs');
  // copy flips
  edits.forEach(function (x) {
    x.targets.forEach(function (f) {
      var s = rd(f);
      if (s.indexOf(x.e.old) === -1) return;
      wr(f, s.split(x.e.old).join(x.e.new));
    });
  });
  // frozen PDF: from the swapped supporter-terms.html
  Object.keys(files).forEach(function (f) { fs.writeFileSync(path.join(root, f), files[f]); });
  var pdf = require('./build-supporter-terms-pdf.js').buildPdf({ root: root, date: date });

  // test pins
  var newCopy = {};
  edits.forEach(function (x) { newCopy[x.e.id] = x.e.new; });
  cfg.flips.forEach(function (fl) {
    fl.pins.forEach(function (p) {
      var nw = p.new.replace(/\{\{new:([a-z0-9-]+)\}\}/g, function (m, id) { if (newCopy[id] === undefined) throw new Error('pin refers to unknown edit ' + id); return jsq(newCopy[id]); });
      wr(p.file, rd(p.file).replace(p.old, function () { return nw; }));
    });
  });
  var lk = rd(LOCK);
  var linkJson = JSON.stringify(links, null, 2).replace(/\n/g, '\n');
  var helper = [
    '// GOLIVE (written by scripts/golive-swap.js): the ONLY Payment Links on the site, pinned exactly. Anything else buy-like still fails.',
    'var GOLIVE_LINKS = ' + linkJson + ';',
    "var GOLIVE_HOST = ['buy', 'stripe', 'com'].join('.');",
    'function goliveHostOk(u) {',
    '  var p;',
    '  try { p = new URL(u); } catch (e) { return false; }',
    "  return p.protocol === 'https:' && p.hostname === GOLIVE_HOST && !p.username && !p.password && !p.port && !p.search && !p.hash &&",
    "    /^\\/[A-Za-z0-9]{6,64}$/.test(p.pathname) && !/test_/i.test(u);",
    '}',
    'function goliveStrip(f, text) {',
    '  var pinned = GOLIVE_LINKS[f];',
    '  if (!pinned) return text;',
    '  Object.keys(pinned).forEach(function (id) {',
    "    assert(goliveHostOk(pinned[id]), id + ' must be an https Payment Link on the Stripe host: plain path, no userinfo, no test marker');",
    "    var anchor = '<a class=\"rt-btn-ghost\" href=\"' + pinned[id] + '\" rel=\"noopener\" target=\"_blank\" data-golive-link=\"' + id + '\"';",
    "    assert(text.split(anchor).length === 2, f + ' must carry ' + id + ' exactly once, exactly as pinned');",
    "    text = text.replace(anchor, '<a class=\"rt-btn-ghost\" data-golive-link=\"' + id + '\"');",
    '  });',
    '  return text;',
    '}',
    ''
  ].join('\n');
  lk = lk.replace(lockOld.strip, function () { return "  var text = goliveStrip(f, read(f));\n  if (f.indexOf('workers/') !== 0) {"; });
  lk = lk.replace(lockOld.siteCount, function () { return helper + lockOld.siteCount; });
  var nLic = Object.keys(links['licence.html'] || {}).length;
  var nIdx = Object.keys(links['index.html'] || {}).length;
  lk = lk.replace(lockOld.l119, function () { return "assert((licence.match(/data-golive-link=/g) || []).length === " + nLic + " && !/data-checkout-test-link|TEST_PAYMENT_LINK/.test(licence), 'licence.html: exactly the pinned buy links, no placeholder left');\n"; });
  lk = lk.replace(lockOld.l127, function () { return "assert(tos.indexOf('last updated " + date + "') !== -1 && tos.indexOf('GO-LIVE MERGE DATE') === -1, 'terms-of-sale date is the go-live date');\n"; });
  lk = lk.replace(lockOld.l152, function () { return "assert(!/TEST_PAYMENT_LINK/.test(licence), 'no checkout placeholder left in licence.html');\n"; });
  lk = lk.replace(lockOld.sp, function () { return "assert((sp.match(/data-golive-link=/g) || []).length === " + nIdx + " && !/data-checkout-test-link|TEST_PAYMENT_LINK/.test(sp), 'supporter section: exactly the pinned buy links, no placeholder left');\n"; });
  lk = lk.replace(lockOld.l201, function () { return "assert(!/\\bhref=\"(?!mailto:|tel:|\\/supporter-terms\\.html|\\/privacy\\.html|https:\\/\\/rathor\\.ai)/.test(goliveStrip('index.html', sp)), 'supporter section links only to mailto, tel, the terms and privacy pages, plus the pinned buy links');\n"; });
  lk = lk.replace(lockOld.l210, function () { return "assert(sterms.indexOf('Version 1.0 · Effective " + date + "') !== -1 && sterms.indexOf('GO-LIVE MERGE DATE') === -1, 'supporter terms carry the go-live date');\n"; });
  var inv = V.DISCORD_INVITE;
  lk = lk.replace(lockOld.l211, function () { return "assert(sterms.indexOf('Join the server with this link: <a href=\"" + inv + "\" rel=\"noopener\" target=\"_blank\">" + inv + "</a>') !== -1, 'supporter terms carry the confirmed Discord invite');\n"; });
  lk = lk.replace(lockOld.l218, function () {
    return [
      "var GOLIVE_DISCORD = '" + inv + "';",
      "(function () { var p = new URL(GOLIVE_DISCORD); assert(p.protocol === 'https:' && !p.username && !p.password && !p.port && !p.search && !p.hash && (p.hostname === 'discord.gg' || (p.hostname === 'discord.com' && /^\\/invite\\//.test(p.pathname))), 'pinned Discord invite shape'); })();",
      "assert(((sp + sterms + supporters + priv).match(/https?:\\/\\/[^\\s\"'<>]*discord[^\\s\"'<>]*/gi) || []).every(function (u) { return u === GOLIVE_DISCORD || u === 'https://discord.com/privacy'; }), 'no Discord link in the supporter pages other than the pinned invite (and the Discord privacy policy link)');",
      ''
    ].join('\n');
  });
  wr(LOCK, lk);
  // sha pins
  Object.keys(files).forEach(function (f) { fs.writeFileSync(path.join(root, f), files[f]); });
  var ART = 'tests/art-uplift-3.test.js';
  var art = rd(ART);
  ['index.html', 'licence.html'].forEach(function (f) {
    var re = new RegExp('(  "' + f.replace('.', '\\.') + '": ")[0-9a-f]{64}(")');
    if (!re.test(art)) throw new Error('art-uplift-3 pin for ' + f + ' not found');
    art = art.replace(re, function (m, a, b) { return a + sha(fs.readFileSync(path.join(root, f))) + b; });
  });
  wr(ART, art);
  var FI = 'tests/footer-i18n-1.test.js';
  var fi = rd(FI);
  packNames.forEach(function (n) {
    var re = new RegExp("(  '" + n.replace('.', '\\.') + "': ')[0-9a-f]{64}(')");
    if (!re.test(fi)) throw new Error('footer-i18n-1 pin for ' + n + ' not found');
    fi = fi.replace(re, function (m, a, b) { return a + sha(fs.readFileSync(path.join(root, 'i18n', n))) + b; });
  });
  wr(FI, fi);
  Object.keys(files).forEach(function (f) { fs.writeFileSync(path.join(root, f), files[f]); });
  console.log('WROTE ' + Object.keys(files).filter(function (f) { return true; }).length + ' files read/updated; terms PDF: ' + path.relative(root, pdf.out) + ' sha256 ' + pdf.sha256);
  console.log('Next: node tests/run-node-tests.js; git add -u + the PDF (never docs/whitepapers/*.pdf); push.');
}

module.exports = { validPayLink: validPayLink, validInvite: validInvite, validDate: validDate };

if (require.main === module) {
  try { main(); } catch (e) { console.error('ERROR: ' + e.message); process.exit(1); }
}
