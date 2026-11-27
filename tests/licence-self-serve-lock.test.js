/* LICENCE-SELF-SERVE-1: self-serve terms and prices are published, but NOTHING is live.
 * - No Stripe checkout / Payment Link / Stripe.js URL in any top-level page, js/ or i18n/ file.
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
var live = /checkout\.stripe\.com|buy\.stripe\.com|js\.stripe\.com|pk_live_|sk_live_|plink_/i;
var files = fs.readdirSync(root).filter(function (n) { return /\.html$/.test(n); });
['js', 'i18n'].forEach(function (d) {
  fs.readdirSync(path.join(root, d)).filter(function (n) { return /\.(js|json)$/.test(n); }).forEach(function (n) { files.push(d + '/' + n); });
});
files.forEach(function (f) { assert(!live.test(read(f)), f + ' must not carry a live Stripe checkout link or key'); });
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
assert(!/(\+?1[ .-]?)?\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}/.test(tos + licence + read('contact.html')), 'no phone number until Sherif chooses one');
assert(licence.indexOf('[SHERIF TO CHOOSE PUBLIC BUSINESS EMAIL/PHONE]') !== -1, 'phone placeholder stays marked');
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
assert(priv.indexOf('Browsing this website and using Lattice Chat collect no personal data.') !== -1, 'privacy browsing claim');
['Accountability.', 'What we collect.', 'Why.', 'Consent.', 'Where it is processed.', 'How long we keep it.', 'Safeguards.', 'Access and correction.', 'Questions and complaints.'].forEach(function (h) {
  assert(priv.indexOf('<strong class="text-amber-300">' + h + '</strong>') !== -1, 'privacy Licence purchases must cover PIPEDA point: ' + h);
});
assert(priv.indexOf('https://www.priv.gc.ca') !== -1, 'privacy must name the OPC');
console.log('licence-self-serve-lock: nothing live, two tiers, no auto-renew, privacy truthful');
