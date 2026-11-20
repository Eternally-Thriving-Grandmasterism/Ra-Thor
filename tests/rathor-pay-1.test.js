/* RATHOR-PAY-1: Support Ra-Thor (Stripe TEST mode, draft only).
 * - Only test-mode buy.stripe.com/test_ links appear in the site pages.
 * - Each Supporter option is one whole-card anchor with no nested link.
 * - The Commercial Licence buy entry is in the markup but not an active href.
 * - No published add-on prices.
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
assert((sec.match(/<a[\s>]/g) || []).length === 2, '#support-rathor must hold exactly two links (the two cards)');
assert(sec.indexOf('Ra-Thor stays free for personal, education and research use.') !== -1, 'free-use line must stay');
assert(sec.indexOf('No features are locked.') !== -1, 'no-features-locked line must stay');
assert(sec.indexOf('Payments are processed by Stripe for Autonomicity Games Inc.') !== -1, 'payee line must stay');

// Licence buy entry: present, disabled, not an active href.
var licenceRaw = read('licence.html');
var licence = stripComments(licenceRaw);
assert(/<section id="licence-buy"/.test(licence), 'licence.html must contain #licence-buy');
assert(/data-rt-licence-buy="off"/.test(licence), 'licence buy entry must stay off until counsel answers');
assert(/aria-disabled="true"/.test(licence), 'licence buy entry must be aria-disabled');
assert(licence.indexOf('href="' + LICENCE + '"') === -1, 'licence Stripe URL must not be an active href');
assert(!/href="https:\/\/buy\.stripe\.com/.test(licence), 'licence.html must have no active Stripe href');
assert(licence.indexOf('data-stripe-test-url="' + LICENCE + '"') !== -1, 'licence test URL kept in data attribute');

// No published add-on prices: the only price on the licence entry is the base.
var buy = licence.match(/<section id="licence-buy"[\s\S]*?<\/section>/)[0];
var prices = buy.match(/\$\s?\d[\d,.]*/g) || [];
assert(prices.length === 1 && prices[0] === '$100', 'licence entry must show only the $100 base price, found ' + prices.join(', '));
assert(/Add-ons are quoted by email at info@Rathor\.ai\./.test(buy), 'add-ons must be quoted by email');

console.log('RATHOR-PAY-1: test-mode links only, whole-card support anchors, licence buy disabled, no add-on prices');
