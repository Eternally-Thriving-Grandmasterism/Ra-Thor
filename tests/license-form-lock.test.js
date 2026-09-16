/* LICENSE-FORM-1: commercial inquiry mailto. No checkout. Family walk unchanged. */
var fs = require('fs');
var path = require('path');
var root = path.join(__dirname, '..');
var inquiry = require('../js/license-inquiry.js');

function assert(cond, msg) {
  if (!cond) throw new Error(msg);
}

var contactHtml = fs.readFileSync(path.join(root, 'contact.html'), 'utf8');
var privacyHtml = fs.readFileSync(path.join(root, 'privacy.html'), 'utf8');
var familyNav = fs.readFileSync(path.join(root, 'js/family-nav-2026-08-22.js'), 'utf8');
var inquiryJs = fs.readFileSync(path.join(root, 'js/license-inquiry.js'), 'utf8');

assert(contactHtml.indexOf('id="license-inquiry-form"') !== -1, 'contact.html must host the commercial inquiry form');
assert(contactHtml.indexOf('id="commercial-inquiry"') !== -1, 'contact.html must keep #commercial-inquiry');
assert(contactHtml.indexOf('name="organization"') !== -1, 'form must include organization name');
assert(contactHtml.indexOf('name="url"') !== -1, 'form must include public site / product URL');
assert(contactHtml.indexOf('wrapping a model') !== -1, 'form must offer wrapping a model');
assert(contactHtml.indexOf('shipping a product') !== -1, 'form must offer shipping a product');
assert(contactHtml.indexOf('internal eval') !== -1, 'form must offer internal eval');
assert(contactHtml.indexOf('name="use"') !== -1 && contactHtml.indexOf('value="other"') !== -1, 'form must offer other');
assert(contactHtml.indexOf('name="seats"') !== -1, 'form must include approx seats or servers');
assert(contactHtml.indexOf('name="message"') !== -1, 'form must include message');
assert(contactHtml.indexOf('Personal / education / modest freelance remains free under AG-SML v1.1.') !== -1,
  'visible note must keep personal / education / modest freelance free');
assert(contactHtml.indexOf('This form is for organization or revenue-generating use.') !== -1,
  'visible note must name organization or revenue-generating use');
assert(contactHtml.indexOf('A human replies.') !== -1, 'visible note must say a human replies');
assert(contactHtml.indexOf('There is no instant license key and no public checkout.') !== -1,
  'visible note must refuse instant key and public checkout');

assert(inquiry.MAILTO === 'info@Rathor.ai', 'mailto target must be info@Rathor.ai');
assert(inquiry.SUBJECT === 'AG-SML commercial inquiry', 'subject must be AG-SML commercial inquiry');

var href = inquiry.encodeMailto({
  organization: 'Acme Lattice',
  url: 'https://acme.example',
  use: 'wrapping a model',
  seats: '12 servers',
  message: 'We wrap an on-device model.'
});
assert(href.indexOf('mailto:info@Rathor.ai?') === 0, 'mailto must target info@Rathor.ai');
assert(href.indexOf('subject=' + encodeURIComponent('AG-SML commercial inquiry')) !== -1,
  'mailto subject must encode AG-SML commercial inquiry');
assert(href.indexOf(encodeURIComponent('Organization name: Acme Lattice')) !== -1, 'body must include organization');
assert(href.indexOf(encodeURIComponent('Public site / product URL: https://acme.example')) !== -1, 'body must include URL');
assert(href.indexOf(encodeURIComponent('Use: wrapping a model')) !== -1, 'body must include use');
assert(href.indexOf(encodeURIComponent('Approx seats or servers: 12 servers')) !== -1, 'body must include seats');
assert(href.indexOf(encodeURIComponent('We wrap an on-device model.')) !== -1, 'body must include message');

var forbidden = /stripe|formspree|wallet|checkout\.stripe|js\.stripe/i;
assert(!forbidden.test(inquiryJs), 'license-inquiry.js must not name a payment processor');
assert(!forbidden.test(contactHtml), 'contact.html must not name a payment processor');
assert(inquiryJs.indexOf('document.cookie') === -1, 'license-inquiry.js must not set cookies');
assert(inquiryJs.indexOf('fetch(') === -1, 'license-inquiry.js must not fetch a backend');
assert(inquiryJs.indexOf('XMLHttpRequest') === -1, 'license-inquiry.js must not post via XHR');
assert(inquiryJs.indexOf('localStorage') === -1, 'license-inquiry.js must not persist the form');

assert(privacyHtml.indexOf('does not upload to rathor.ai') !== -1, 'privacy.html must say the form does not upload');
assert(privacyHtml.indexOf('opens your mail app') !== -1, 'privacy.html must say it opens your mail app');

var linkBlock = familyNav.slice(familyNav.indexOf('var LINKS = ['), familyNav.indexOf('];', familyNav.indexOf('var LINKS = [')) + 2);
assert(linkBlock.indexOf("{ href: '/', label: 'Home' }") !== -1, 'family walk must keep Home');
assert(linkBlock.indexOf("{ href: '/chat.html', label: 'Chat' }") !== -1, 'family walk must keep Chat');
assert(linkBlock.indexOf("{ href: '/employ.html', label: 'Employ' }") !== -1, 'family walk must keep Employ');
assert(linkBlock.indexOf("{ href: '/Launch-Ra-Thor.html', label: 'Launch' }") !== -1, 'family walk must keep Launch');
assert(linkBlock.indexOf("{ href: '/micro-moment.html', label: 'Moments' }") !== -1, 'family walk must keep Moments');
assert(linkBlock.indexOf("{ href: '/sovereign-shard.html', label: 'Shard' }") !== -1, 'family walk must keep Shard');
assert(linkBlock.indexOf("{ href: '/web-forge.html', label: 'Forge' }") !== -1, 'family walk must keep Forge');
assert(linkBlock.indexOf("{ href: '/contact.html', label: 'Contact' }") !== -1, 'family walk must keep Contact');
assert(linkBlock.indexOf("{ href: '/privacy.html', label: 'Privacy' }") !== -1, 'family walk must keep Privacy');
assert((linkBlock.match(/href:/g) || []).length === 9, 'family walk must stay nine destinations');
assert(linkBlock.indexOf('license') === -1, 'family walk must not grow a license page');

assert(contactHtml.indexOf('inspect ≠ METR') === -1, 'contact form must not invent METR copy');
assert(inquiryJs.indexOf('AGSi') === -1, 'inquiry script must not claim Combined AGSi');

console.log('LICENSE-FORM-1 license-form-lock checks passed');
