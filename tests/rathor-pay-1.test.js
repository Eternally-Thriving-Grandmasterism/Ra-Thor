/* RATHOR-PAY-1: Support Ra-Thor (Stripe TEST mode, draft only).
 * - Only test-mode buy.stripe.com/test_ links appear in the site pages.
 * - Each Supporter option is one whole-card anchor with no nested link.
 * - The Commercial Licence buy entry is in the markup but not an active href.
 * - No price at all on licence.html; Supporter prices carry a USD label.
 * - The old 'no personal data whatsoever' line is gone from English sources.
 */
var fs = require('fs');
var path = require('path');
var root = path.join(__dirname, '..');

function assert(cond, msg) {
  if (!cond) throw new Error('rathor-pay-1: ' + msg);
}
function read(rel) {
  return fs.readFileSync(path.join(root, rel), 'utf8');
}
function stripComments(html) {
  return html.replace(/<!--[\s\S]*?-->/g, '');
}

var MONTHLY = 'https://buy.stripe.com/test_6oU28qfKm2c5evZ3s4aAw0e';
var YEARLY = 'https://buy.stripe.com/test_fZu6oGfKm7wp9bF6EgaAw0f';
var LICENCE = 'https://buy.stripe.com/test_00w9AS55IeYR9bFgeQaAw0g';

var pages = ['index.html', 'licence.html', 'privacy.html', 'contact.html', 'pilot.html', 'employ.html'];
pages.forEach(function (rel) {
  var html = read(rel);
  var stripe = html.match(/https?:\/\/(?:buy|checkout)\.stripe\.com\/[^"'\s<>]*/g) || [];
  stripe.forEach(function (u) {
    assert(/^https:\/\/buy\.stripe\.com\/test_[A-Za-z0-9]+$/.test(u), rel + ': non-test Stripe link ' + u);
  });
  assert(!/\b(?:pk|sk|rk)_(?:live|test)_[A-Za-z0-9]/.test(html), rel + ': Stripe key-like string present');
});

// Support section: two whole-card anchors, no nested links.
var index = stripComments(read('index.html'));
var secMatch = index.match(/<section id="support-rathor"[\s\S]*?<\/section>/);
assert(secMatch, 'index.html must contain #support-rathor');
var sec = secMatch[0];
[MONTHLY, YEARLY].forEach(function (url) {
  var re = new RegExp('<a href="' + url.replace(/[.*+?^${}()|[\]\\\/]/g, '\\$&') + '"([^>]*)>([\\s\\S]*?)</a>');
  var m = sec.match(re);
  assert(m, 'support card for ' + url + ' must be an <a> card');
  assert(/\brt-card-uniform\b/.test(m[1]) && /\bblock\b/.test(m[1]), 'support card must be a block rt-card-uniform anchor');
  assert(/rel="noopener"/.test(m[1]), 'support card must carry rel="noopener"');
  assert(!/<a[\s>]/.test(m[2]), 'support card must not contain a nested link');
});
var secLinks = sec.match(/<a [^>]*>/g) || [];
var nonMailto = secLinks.filter(function (t) { return !/href="mailto:/.test(t); });
assert(nonMailto.length === 2, '#support-rathor must hold exactly two non-mailto links (the two cards)');
assert(sec.indexOf('Supporter · $5 USD / month') !== -1, '$5 Supporter price must carry its USD label');
assert(sec.indexOf('Supporter · $50 USD / year') !== -1, '$50 Supporter price must carry its USD label');
var supPrices = sec.match(/\$\s?\d[\d,.]*(?![\d,.])(?!\s+USD)/g) || [];
assert(supPrices.length === 0, 'every Supporter price must be followed by USD, found ' + supPrices.join(', '));
assert(sec.indexOf('Want your name on a supporters list? Email <a href="mailto:info@Rathor.ai"') !== -1, 'supporters-list email line must be present');
assert(sec.indexOf('opt in') === -1, 'old opt-in supporters wording must be gone');
assert(sec.indexOf('Ra-Thor stays free for personal, education and research use.') !== -1, 'free-use line must stay');
assert(sec.indexOf('No features are locked.') !== -1, 'no-features-locked line must stay');
assert(sec.indexOf('Payments are processed by Stripe for Autonomicity Games Inc.') !== -1, 'payee line must stay');

// Licence buy entry: present, disabled, not an active href, and no price anywhere on licence.html.
var licenceRaw = read('licence.html');
var licence = stripComments(licenceRaw);
assert(/<section id="licence-buy"/.test(licence), 'licence.html must contain #licence-buy');
assert(/data-rt-licence-buy="off"/.test(licence), 'licence buy entry must stay off until counsel answers');
assert(/aria-disabled="true"/.test(licence), 'licence buy entry must be aria-disabled');
assert(licence.indexOf('href="' + LICENCE + '"') === -1, 'licence Stripe URL must not be an active href');
assert(!/href="https:\/\/buy\.stripe\.com/.test(licence), 'licence.html must have no active Stripe href');
assert(licence.indexOf('data-stripe-test-url="' + LICENCE + '"') !== -1, 'licence test URL kept in data attribute');
assert(licence.indexOf('Commercial licence purchase coming soon. Contact info@Rathor.ai') !== -1, 'disabled entry text must match the ruling');
var licPrices = licenceRaw.match(/(?:\$|USD|CAD|€|£)\s?\d|\d[\d,.]*\s?(?:USD|CAD|\/\s?(?:year|yr|month|mo)\b)/g) || [];
assert(licPrices.length === 0, 'licence.html must show no price, found ' + licPrices.join(', '));

// Privacy: the old absolute line is gone from the English sources.
['privacy.html', 'i18n/en.js', 'i18n/en.json', 'index.html', 'js/family-nav-2026-08-22.js',
 'docs/Legal/privacy.md', 'ra-thor-website-1.html'].forEach(function (rel) {
  assert(!/no personal data whatsoever/i.test(read(rel)), rel + ': "no personal data whatsoever" must be gone');
});
var stripeLine = 'Payments on this website are processed by Stripe, which handles card and payment details; we receive the payer’s name, email and payment status. Browsing this website and the offline chat still collect no personal data.';
assert(read('privacy.html').indexOf(stripeLine) !== -1, 'privacy.html must carry the drafted Stripe line');
assert(read('i18n/en.js').indexOf(stripeLine) !== -1, 'i18n/en.js must carry the drafted Stripe line');

console.log('RATHOR-PAY-1: test-mode links only, USD-labelled whole-card support anchors, supporters email line, licence buy disabled with no price, privacy line updated');
