/* GOLIVE-PREP-1: the go-live swap stays small and mechanical.
 * Before the swap (this branch): every one of the 7 values has exactly one token and one marker comment, the five buy buttons
 * are inert spans, the swap script's dry run finds no mismatch, the PDF builder takes the terms text without the notice,
 * and the value validators refuse userinfo, test links, other hosts and plain http.
 * After the swap (state B, written by scripts/golive-swap.js) the same file checks the five anchors instead. */
var fs = require('fs');
var path = require('path');
var cp = require('child_process');
var root = path.join(__dirname, '..');
function assert(c, m) { if (!c) throw new Error(m); }
function read(rel) { return fs.readFileSync(path.join(root, rel), 'utf8'); }
function noComments(s) { return s.replace(/<!--[\s\S]*?-->/g, ''); }
function count(h, n) { return h.split(n).length - 1; }

var cfg = JSON.parse(read('scripts/golive-swap.config.json'));
assert(cfg.swapSpots.length === 7, 'seven swap spots');
assert(cfg.swapSpots.filter(function (s) { return s.kind === 'paylink'; }).length === 5, 'five Payment Link spots');
var ids = cfg.swapSpots.map(function (s) { return s.id; }).sort().join(',');
assert(ids === 'DISCORD_INVITE,GOLIVE_DATE,PAYLINK_100,PAYLINK_400,PAYLINK_SP_12M,PAYLINK_SP_1M,PAYLINK_SP_3M', 'the seven ids');
assert(Object.keys(cfg.labels).length === 5, 'a label slot per buy button');

var swapped = fs.existsSync(path.join(root, 'terms/supporter-pass-v1.0.pdf'));
if (!swapped) {
  cfg.swapSpots.forEach(function (sp) {
    (sp.files || [sp.file]).forEach(function (f) {
      var t = read(f);
      assert(count(noComments(t), sp.token) === 1, sp.id + ': token ' + sp.token + ' exactly once in ' + f);
      assert(count(t, 'GOLIVE-SWAP ' + sp.id + ':') === 1, sp.id + ': one marker comment in ' + f);
    });
  });
  var spans = (read('licence.html') + read('index.html')).match(/<span class="rt-btn-ghost"[^>]*data-checkout-test-link[^>]*>/g) || [];
  assert(spans.length === 5 && spans.every(function (t) { return !/\bhref=/.test(t) && /aria-disabled="true"/.test(t); }), 'five inert buy spans, no href');
  var r = cp.spawnSync(process.execPath, [path.join(root, 'scripts/golive-swap.js')], { encoding: 'utf8' });
  assert(r.status === 0 && !/PROBLEM:/.test(r.stdout), 'swap dry run finds no mismatch: ' + r.stdout.slice(0, 400));
  assert(/NOT READY/.test(r.stdout) && /MISSING: value PAYLINK_100/.test(r.stdout), 'dry run reports the missing values');
  var w = cp.spawnSync(process.execPath, [path.join(root, 'scripts/golive-swap.js'), '--write'], { encoding: 'utf8' });
  assert(w.status === 3 && /refused/.test(w.stdout), '--write refuses while values or copy are missing');
  assert(!fs.existsSync(path.join(root, 'terms/supporter-pass-v1.0.pdf')), 'the refused write must not create the PDF');
} else {
  assert(count(read('licence.html'), 'data-golive-link=') === 2 && count(read('index.html'), 'data-golive-link=') === 3, 'after the swap: the pinned anchors');
}

var sw = require('../scripts/golive-swap.js');
var H = ['buy', 'stripe', 'com'].join('.');
assert(sw.validPayLink('https://' + H + '/aBc123xYz9'), 'plain link accepted');
['http://' + H + '/aBc123xYz9', 'https://' + H + '@evil.example/aBc123xYz9', 'https://user:pw@' + H + '/aBc123xYz9', 'https://' + H + '.evil.example/aBc123xYz9',
 'https://evil.example/' + H + '/aBc123xYz9', 'https://' + H + '/test_aBc123xYz9', 'https://' + H + '/aBc123xYz9?x=1', 'https://' + H + '/aBc123xYz9#x',
 'https://' + H + ':8443/aBc123xYz9', 'https://' + H + '/ab', 'https://' + H + '/a/b/c123456', 'https://checkout.stripe.com/c/pay/abc123456'].forEach(function (u) {
  assert(!sw.validPayLink(u), 'must refuse ' + u);
});
assert(sw.validInvite('https://discord.gg/AbCd123') && sw.validInvite('https://discord.com/invite/AbCd123'), 'invites accepted');
['http://discord.gg/AbCd123', 'https://discord.gg@evil.example/AbCd123', 'https://discord.gg.evil.example/AbCd123', 'https://discord.com/privacy', 'https://discord.gg/AbCd123?x=1'].forEach(function (u) {
  assert(!sw.validInvite(u), 'must refuse ' + u);
});
assert(sw.validDate('2026-12-01') && !sw.validDate('2026-13-01') && !sw.validDate('2026-02-30') && !sw.validDate('1 December 2026'), 'date format');

var pdf = require('../scripts/build-supporter-terms-pdf.js');
var body = pdf.extractTermsHtml(read('supporter-terms.html'));
assert(body.indexOf('1. Who we are.') !== -1 && body.indexOf('12.') !== -1, 'PDF source carries the sections');
assert(!/not offered yet|DRAFT|<!--|class=/.test(body), 'PDF source drops the notice, comments and site classes');
console.log('golive-swap-1: ' + (swapped ? 'swapped state' : 'parked state') + ' ok (7 spots, validators, dry run, PDF source)');
