/* LICENCE-SELF-SERVE-1: self-serve terms and prices are published, but NOTHING is live.
 * - No Stripe checkout / Payment Link / Stripe.js URL in any site file (.html/.js/.json/.md, nested folders included;
 *   the server-side workers/ tree is checked for live keys only), and no live key or live link anywhere in the repo.
 * - licence.html and terms-of-sale.html say online checkout is not open; the checkout control is disabled.
 * - contact.html keeps "There is no instant license key and no public checkout." (license-form-lock).
 * - LICENSE §2 keeps the written-and-signed path next to the electronic path.
 * - privacy.html keeps the browsing claim truthful and carries the Licence purchases section.
 * Go-live (Sherif's yes, separate PR) is the only change allowed to relax the first two checks. */
var fs = require('fs');
var path = require('path');
var root = path.join(__dirname, '..');
function assert(c, m) { if (!c) throw new Error(m); }
function read(rel) { return fs.readFileSync(path.join(root, rel), 'utf8'); }
var SKIP = { '.git': 1, node_modules: 1, target: 1, dist: 1, build: 1, out: 1, coverage: 1, tests: 1 };
function walk(dir, out) {
  fs.readdirSync(path.join(root, dir), { withFileTypes: true }).forEach(function (e) {
    if (SKIP[e.name] || e.isSymbolicLink()) return;
    var rel = dir ? dir + '/' + e.name : e.name;
    if (e.isDirectory()) return walk(rel, out);
    if (/\.(html|js|mjs|cjs|ts|json|md|toml|ya?ml)$/.test(e.name) && fs.statSync(path.join(root, rel)).size < 4000000) out.push(rel);
  });
}
var all = [];
walk('', all);
assert(all.length > 100, 'nested walk found too few files: ' + all.length);
// Site surface (everything except the server-side workers/ tree): no checkout, Payment Link or Stripe.js URL at all.
var siteLink = /checkout\.stripe\.com|buy\.stripe\.com|js\.stripe\.com|\bplink_|\b(?:sk|pk|rk)_live_/i;
// Site surface: no buy / checkout / pay link from any provider (Stripe or not) until go-live.
var anyBuyLink = /https?:\/\/[^\s"'<>)]*(?:checkout|\/buy\b|\bbuy\.|\/pay\b|paypal\.|gumroad\.|lemonsqueezy\.|paddle\.(?:com|net)|square\.link)/i;
// Whole repo: no real live key or live link value (bare prefixes in key-mode checks are fine).
var liveValue = /\b(?:sk|pk|rk)_live_[A-Za-z0-9]{8,}|\bcs_live_[A-Za-z0-9]{8,}|buy\.stripe\.com\/[A-Za-z0-9]{6,}/;
var siteCount = 0;
all.forEach(function (f) {
  var text = read(f);
  if (f.indexOf('workers/') !== 0) {
    siteCount++;
    assert(!siteLink.test(text), f + ' must not carry a Stripe checkout link, Payment Link, Stripe.js or live key prefix');
    assert(!anyBuyLink.test(text), f + ' must not carry a buy / checkout / pay link');
  }
  assert(!liveValue.test(text), f + ' must not carry a live Stripe key or live link');
});
assert(all.some(function (f) { return /\.md$/.test(f) && f.indexOf('/') !== -1; }), 'nested .md files must be scanned');
assert(all.some(function (f) { return /^js\/[^/]+\/.+\.js$/.test(f); }), 'js/ subfolders must be scanned');
var licence = read('licence.html');
var tos = read('terms-of-sale.html');
assert(licence.indexOf('Online checkout is not open yet.') !== -1, 'licence.html must say checkout is not open');
assert(licence.indexOf('aria-disabled="true"') !== -1, 'licence.html checkout control must be disabled');
assert(tos.indexOf('Online checkout is not open yet.') !== -1, 'terms-of-sale.html must say checkout is not open');
assert(licence.indexOf('USD $100 for 12 months') !== -1 && licence.indexOf('USD $400 for 12 months') !== -1, 'two tiers, USD, 12 months');
assert(licence.indexOf('Does not renew automatically.') !== -1, 'no auto-renew line');
// Council default tier limits (Sherif may veto): $100 = 1 org, up to 5 users, under USD $250,000/yr; $400 = 1 org, up to 25 users, under USD $1,000,000/yr.
assert(licence.indexOf('up to 5 users, and under USD $250,000 a year') !== -1 && tos.indexOf('up to 5 users, and under USD $250,000 a year') !== -1, '$100 tier limits');
assert(licence.indexOf('up to 25 users, and under USD $1,000,000 a year') !== -1 && tos.indexOf('up to 25 users, and under USD $1,000,000 a year') !== -1, '$400 tier limits');
assert(!/up to 10 people|one production deployment/.test(tos + licence), 'old 10-people / 1-deployment limits are gone');
assert(tos.indexOf('last updated [GO-LIVE MERGE DATE]') !== -1, 'terms date stays a placeholder until go-live');
assert(tos.indexOf('limited to the fees you paid for this licence') !== -1, 'liability cap = fees paid');
// Public business phones chosen by Sherif (7:25 PM ET 2026-11-26). Exactly these two; any other phone pattern fails.
var PHONES = ['US: <a href="tel:+16504447224" class="text-amber-300 underline">+1 650-444-7224</a>', 'Canada: <a href="tel:+14169047332" class="text-amber-300 underline">+1 416-904-7332</a>'];
var contact = read('contact.html');
[['licence.html', licence], ['terms-of-sale.html', tos], ['contact.html', contact]].forEach(function (pair) {
  PHONES.forEach(function (p) { assert(pair[1].indexOf(p) !== -1, pair[0] + ' must carry ' + p); });
  var rest = pair[1].split(PHONES[0]).join('').split(PHONES[1]).join('');
  assert(!/(\+?1[ .-]?)?\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}/.test(rest), pair[0] + ' carries a phone number other than the two approved ones');
  assert(!/tel:/i.test(rest), pair[0] + ' carries a tel: link other than the two approved ones');
  assert(rest.indexOf('[SHERIF TO CHOOSE PUBLIC BUSINESS EMAIL/PHONE]') === -1, pair[0] + ' must not keep the phone placeholder');
});
[licence, tos].forEach(function (h) { assert(h.indexOf('Autonomicity Games Inc.') !== -1, 'seller is Autonomicity Games Inc.'); });
assert(tos.indexOf('It does not renew and we will not charge you again.') !== -1, 'terms: no auto-renew');
assert(!/lawyer[- ]approved|lawyer[- ]reviewed/i.test(tos + licence), 'never claim lawyer approval');
assert(read('contact.html').indexOf('There is no instant license key and no public checkout.') !== -1, 'contact keeps the no-checkout line');
var lic = read('LICENSE');
assert(lic.indexOf('either written and signed (email to info@Rathor.ai)') !== -1, 'LICENSE §2 keeps written-and-signed path');
assert(lic.indexOf('or formed electronically: acceptance of the Ra-Thor Commercial Licence Terms') !== -1, 'LICENSE §2 electronic path');
assert(tos.indexOf('full refund within 30 days of the payment date') !== -1, 'terms: 30-day refund on request');
assert(!/unused days|unused part/i.test(tos + licence), 'no prorated-refund wording');
assert(tos.indexOf('correct errors, or decline') !== -1, 'terms: accept / decline / correct step (CPA 2002 s.38(2))');
assert(tos.indexOf('within 15 days') !== -1, 'terms: copy within 15 days (O. Reg. 17/05 s.33)');
assert(tos.indexOf('buying for business purposes') !== -1, 'terms: business-purpose tick box');
assert(tos.indexOf('European Union, the United Kingdom, Russia, or sanctioned countries') !== -1, 'terms: EU/UK/Russia/sanctions block');
assert(/\[TEST_PAYMENT_LINK_100\]/.test(licence) && /\[TEST_PAYMENT_LINK_400\]/.test(licence), 'checkout placeholders stay placeholders until go-live');
var priv = read('privacy.html');
assert(priv.indexOf('id="licence-purchases"') !== -1, 'privacy.html must carry Licence purchases');
assert(priv.indexOf('collects no personal data whatsoever') === -1, 'privacy.html must not keep the unqualified claim');
assert(priv.indexOf('The site itself collects no personal data: no sign-in, cookies, analytics or Ra-Thor server. Lattice Chat keeps your sessions in this browser’s localStorage on your device. If you choose an optional third-party service (opening Grok or X, Google Translate, downloading an on-device model, or a model server you configure), that service handles what you send under its own policy.') !== -1, 'privacy browsing claim');
['Accountability.', 'What we collect.', 'Why.', 'Consent.', 'Where it is processed.', 'How long we keep it.', 'Safeguards.', 'Access and correction.', 'Questions and complaints.'].forEach(function (h) {
  assert(priv.indexOf('<strong class="text-amber-300">' + h + '</strong>') !== -1, 'privacy Licence purchases must cover PIPEDA point: ' + h);
});
assert(priv.indexOf('https://www.priv.gc.ca') !== -1, 'privacy must name the OPC');
assert(priv.indexOf('Nothing is retained off-device.') === -1 && priv.indexOf('features; the rest of the site requires zero data') === -1, 'privacy rights lines must account for licence purchases');
assert(licence.indexOf('see the fixed 12-month prices above') !== -1 && licence.indexOf('see the fixed 12-month prices below') === -1, 'licence ask text points at the card above');
// Placeholders that must stay visible until Sherif / the accountant fill them at go-live.
assert(tos.indexOf('[BUSINESS ADDRESS — SHERIF TO CHOOSE]') !== -1, 'address placeholder stays until go-live');
assert(tos.indexOf('[TAX LINE — pending accountant;') !== -1, 'tax-line placeholder stays until the accountant answers');
assert(tos.indexOf('make up the whole agreement between us') !== -1, 'terms 19 grammar');
assert(tos.indexOf('this does not include offering Ra-Thor itself to others as hosting, a SaaS or an API') !== -1, 'terms 4 own-products clause excludes SaaS / hosting / API for others');
assert(priv.indexOf('that service handles what you send under its own policy') !== -1, 'privacy names optional third-party services');
assert(priv.indexOf('Lattice Chat collect no personal data') === -1, 'privacy must not claim Lattice Chat collects nothing regardless of chosen services');
console.log('licence-self-serve-lock: nothing live (' + all.length + ' files scanned, ' + siteCount + ' site files), two tiers, no auto-renew, privacy truthful');
