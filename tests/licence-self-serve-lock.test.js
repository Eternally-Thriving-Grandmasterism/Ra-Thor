/* LICENCE-SELF-SERVE-1: self-serve terms and prices are published, but NOTHING is live.
 * - No Stripe checkout / Payment Link / Stripe.js URL in any site file (.html/.js/.json/.md, nested folders included;
 *   the server-side workers/ tree is checked for live keys only), and no live key or live link anywhere in the repo.
 * - licence.html and terms-of-sale.html say online checkout is not open; the checkout control is disabled.
 * - contact.html keeps "There is no instant license key and no public checkout." (license-form-lock).
 * - LICENSE §2 keeps the written-and-signed path next to the electronic path.
 * - privacy.html keeps the browsing claim truthful and carries the Licence purchases and Supporter Pass sections.
 * - Provider documentation / support hosts (docs.stripe.com etc.) may be linked; checkout and buy links may not.
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
// Provider documentation and support hosts only explain how things work; links to them are not buy links.
// Exact host match (no subdomain or suffix tricks). checkout.stripe.com / buy.stripe.com are never on this list.
var DOC_HOSTS = { 'docs.stripe.com': 1, 'support.stripe.com': 1, 'developer.paypal.com': 1, 'developer.paddle.com': 1, 'docs.lemonsqueezy.com': 1, 'help.gumroad.com': 1, 'developer.squareup.com': 1 };
var urlRe = /https?:\/\/[^\s"'<>)]*/gi;
// Buy / checkout hosts that must never appear anywhere inside a URL (userinfo, path, query, fragment), even under a docs host.
var embeddedBuyHost = /(?:checkout|buy|js)\.stripe\.com|paypal\.(?:com|me)|gumroad\.com|lemonsqueezy\.com|paddle\.(?:com|net)|square\.link/i;
function decodeAll(s) { for (var i = 0; i < 3; i++) { try { var d = decodeURIComponent(s); if (d === s) break; s = d; } catch (e) { break; } } return s; }
// Core AMEND 7230505251: the host is new URL(u).hostname, so userinfo (docs.stripe.com:x@evil.tld) cannot pose as a docs host.
function hostOf(u) { try { return new URL(u).hostname.toLowerCase(); } catch (e) { return ''; } }
// Decode the HTML entities that can hide a scheme or slashes (https&#58;// , &#x3a; , &colon; , &#47; , &sol; , &amp;).
function decodeEntities(t) {
  return t.replace(/&#x0*([0-9a-f]+);?/gi, function (m, h) { return String.fromCharCode(parseInt(h, 16)); })
    .replace(/&#0*([0-9]+);?/g, function (m, d) { return String.fromCharCode(parseInt(d, 10)); })
    .replace(/&colon;/gi, ':').replace(/&sol;/gi, '/').replace(/&period;/gi, '.').replace(/&amp;/gi, '&');
}
// Protocol-relative links (//host/...) after a quote, =, ( or space are treated as https.
var protoRelRe = /(^|[\s"'(=])\/\/((?:[a-z0-9-]+\.)+[a-z]{2,}[^\s"'<>)]*)/gi;
function urlIsBuy(u, depth) {
  var p;
  try { p = new URL(u); } catch (e) { return anyBuyLink.test(u); }
  var host = hostOf(u);
  // Everything except the hostname, decoded: userinfo, port, path, query and fragment.
  var rest = decodeAll(p.username + ':' + p.password + '@' + p.port + p.pathname + p.search + p.hash);
  if (embeddedBuyHost.test(rest)) return true;
  if (depth < 3 && (rest.match(urlRe) || []).some(function (n) { return urlIsBuy(n, depth + 1); })) return true;
  if (DOC_HOSTS[host]) return false;
  return anyBuyLink.test(u) || embeddedBuyHost.test(host);
}
function hasBuyLink(text) {
  var t = decodeEntities(text).replace(protoRelRe, function (m, pre, rest) { return pre + 'https://' + rest; });
  return (t.match(urlRe) || []).some(function (u) { return urlIsBuy(u, 0); });
}
// Self-checks for the allowlist.
assert(!hasBuyLink('see https://docs.stripe.com/tax/checkout/tax-ids'), 'docs.stripe.com is documentation, not a buy link');
assert(!hasBuyLink('see https://support.stripe.com/questions/checkout-help'), 'support.stripe.com is documentation, not a buy link');
assert(hasBuyLink('pay at https://buy.stripe.com/test_abc123'), 'buy.stripe.com must stay blocked');
assert(hasBuyLink('pay at https://checkout.stripe.com/c/pay/cs_test_x'), 'checkout.stripe.com must stay blocked');
assert(hasBuyLink('https://docs.stripe.com:x@example.com/checkout'), 'userinfo trick must stay blocked');
assert(hasBuyLink('https://docs.stripe.com.example.com/checkout'), 'look-alike doc host must stay blocked');
assert(hasBuyLink('https://docs.stripe.com/x https://example.com/buy'), 'a real buy link next to a doc link must stay blocked');
assert(hasBuyLink('https://www.paypal.com/checkoutnow?token=x') && hasBuyLink('https://example.gumroad.com/l/x'), 'other providers stay blocked');
assert(hasBuyLink('https://docs.stripe.com:443@evil.example/checkout') && hasBuyLink('https://docs.stripe.com@buy.stripe.com/x'), 'userinfo trick (docs host before @) stays blocked');
assert(hasBuyLink('https://docs.stripe.com/x?u=https://buy.stripe.com/abc') && hasBuyLink('https://docs.stripe.com/x?u=https%3A%2F%2Fcheckout.stripe.com%2Fc'), 'buy/checkout host nested in a docs URL query stays blocked');
assert(hasBuyLink('https://docs.stripe.com/x?next=https://example.com/pay'), 'buy link nested in a docs URL query stays blocked');
assert(hasBuyLink('<a href="//buy.stripe.com/abc">') && hasBuyLink("href='//example.com/checkout'"), 'protocol-relative buy links stay blocked');
assert(!hasBuyLink('<a href="//docs.stripe.com/tax/checkout/tax-ids">') && !hasBuyLink('// checkout comment in code'), 'protocol-relative docs link and code comments pass');
assert(hasBuyLink('https&#58;//buy.stripe.com/abc') && hasBuyLink('https&#x3a;&#x2f;&#x2f;checkout.stripe.com/c') && hasBuyLink('https&colon;//example.com/pay'), 'entity-encoded schemes stay blocked');
assert(!hasBuyLink('https&#58;//docs.stripe.com/tax/checkout/tax-ids'), 'entity-encoded docs link passes');
// Site files: no test-mode key values either (mirrors the live-key check). Dummies are built by concatenation.
var testKeyValue = /\b(?:sk|pk|rk|cs)_test_[A-Za-z0-9]{8,}/;
assert(testKeyValue.test('sk' + '_test_' + 'abcd1234EFGH') && testKeyValue.test('cs' + '_test_' + 'a1b2c3d4e5') && !testKeyValue.test('sk' + '_test_' + 'x'), 'test-key guard self-check');
// Whole repo: no real live key or live link value (bare prefixes in key-mode checks are fine).
var liveValue = /\b(?:sk|pk|rk)_live_[A-Za-z0-9]{8,}|\bcs_live_[A-Za-z0-9]{8,}|buy\.stripe\.com\/[A-Za-z0-9]{6,}/;
var siteCount = 0;
all.forEach(function (f) {
  var text = read(f);
  if (f.indexOf('workers/') !== 0) {
    siteCount++;
    assert(!siteLink.test(text), f + ' must not carry a Stripe checkout link, Payment Link, Stripe.js or live key prefix');
    assert(!hasBuyLink(text), f + ' must not carry a buy / checkout / pay link');
    assert(!testKeyValue.test(text), f + ' must not carry a Stripe test key or test session value');
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
// Seller address and tax line are filled; the go-live date placeholder (asserted above) stays until go-live.
assert(tos.indexOf('<a href="mailto:info@Rathor.ai" class="text-amber-300 underline">info@Rathor.ai</a> · 73 Kingmount Cres, Richmond Hill, ON, Canada L4B 3X4.<br>') !== -1, 'terms 1 seller business address (provided by Sherif, Nov 27, 2026)');
assert(tos.indexOf('[BUSINESS ADDRESS') === -1, 'address placeholder is gone');
assert(tos.indexOf('paid once for 12 months. No GST/HST is charged at checkout at this time. If that changes, GST/HST will be added at checkout.</p>') !== -1, 'terms 10 neutral GST/HST line');
assert(tos.indexOf('small supplier') === -1 && tos.indexOf('C$30,000') === -1, 'terms carry no small-supplier claim (accountant to confirm)');
assert(tos.indexOf('[TAX LINE') === -1, 'tax-line placeholder is gone');
assert(tos.indexOf('make up the whole agreement between us') !== -1, 'terms 19 grammar');
assert(tos.indexOf('this does not include offering Ra-Thor itself to others as hosting, a SaaS or an API') !== -1, 'terms 4 own-products clause excludes SaaS / hosting / API for others');
assert(priv.indexOf('that service handles what you send under its own policy') !== -1, 'privacy names optional third-party services');
// Supporter Pass privacy (v5).
assert(priv.indexOf('id="supporter-pass"') !== -1, 'privacy.html must carry the Supporter Pass section');
assert(priv.indexOf('Supporter Passes are not offered yet. This section applies only once they are, and only if you buy one.') !== -1, 'supporter section: not offered yet, scoped to buyers');
assert(priv.indexOf('We receive the purchase records Stripe provides: your name, email, country, the amount and the date.') !== -1, 'supporter purchase records listed');
assert(priv.indexOf('It does not have to be your real name. If you leave it blank, you are not listed. Nothing else about you is shown publicly: no email, country, amount or date.') !== -1, 'supporter display name is optional and the only public detail');
assert(priv.indexOf('Email info@Rathor.ai at any time to have your name removed. We also remove it within 7 days of a refund or of the end of your pass.') !== -1, 'supporter name removal');
assert(priv.indexOf('earlier versions in that repository’s history may still show a name after we remove it') !== -1, 'supporter repo-history note');
assert(priv.indexOf('We do not use Supporter Pass records for marketing, and we do not sell your information.') !== -1, 'supporter no marketing / no selling');
assert(priv.indexOf('You can ask to see or correct your information, including your Supporter Pass records, by emailing info@Rathor.ai.') !== -1 && priv.indexOf('Email info@Rathor.ai, including about your Supporter Pass records.') !== -1, 'shared access and complaints lines cover Supporter Pass records');
assert(priv.indexOf('We use it only to give you the role and to remove it within 7 days of a refund or of the end of your pass. It is also kept') !== -1, 'privacy names the optional Discord username and its limited use');
// SUPPORTER-PASS-1: Discord username and Stripe-record retention wording (Legal Mate 03 G); display name carries the Stripe-record clause too.
assert(priv.indexOf('<strong class="text-amber-300">Optional Discord username.</strong>') !== -1 && priv.indexOf('https://discord.com/privacy') !== -1, 'privacy: Optional Discord username bullet with Discord privacy link');
assert(priv.indexOf('It is also kept with your Stripe purchase record, which we keep only as long as tax and accounting law requires.') !== -1, 'privacy: Discord username also kept in the Stripe purchase record');
assert(priv.indexOf('Your display name: only while it is listed on the supporters page. A copy also stays with your Stripe purchase record, which we keep only as long as tax and accounting law requires. Your Discord username: in our own working list only until your role is removed; it also stays in the Stripe purchase record for as long as that record is kept.') !== -1, 'privacy: retention bullet covers display name and Discord username with the Stripe-record clause');
var privMd = read('docs/Legal/privacy.md');
assert(privMd.indexOf('- **Optional Discord username.**') !== -1 && privMd.indexOf('We use it only to give you the role and to remove it within 7 days of a refund or of the end of your pass. It is also kept') !== -1, 'privacy.md mirrors the Discord username bullet');
assert(privMd.indexOf('Your display name: only while it is listed on the supporters page. A copy also stays with your Stripe purchase record, which we keep only as long as tax and accounting law requires. Your Discord username: in our own working list only until your role is removed;') !== -1, 'privacy.md mirrors the retention bullet');
assert(priv.indexOf('Supporter Passes are not offered yet. This section applies only once they are') !== -1, 'privacy: not-offered-yet line stays until Sherif goes live');

// SUPPORTER-PASS-1: three one-time passes (USD 5 / 12 / 35), checkout inert, Stripe TEST only, placeholders stay placeholders.
var home = read('index.html');
var spMatch = home.match(/<section id="supporter-pass"[\s\S]*?<\/section>/);
assert(spMatch, 'index.html must carry #supporter-pass');
var sp = spMatch[0];
['[TEST_PAYMENT_LINK_SP_1M]', '[TEST_PAYMENT_LINK_SP_3M]', '[TEST_PAYMENT_LINK_SP_12M]'].forEach(function (ph) {
  assert(sp.indexOf('data-checkout-test-link="' + ph + '"') !== -1, 'supporter pass placeholder ' + ph);
});
var spans = sp.match(/<span class="rt-btn-ghost"[^>]*>/g) || [];
assert(spans.length === 3 && spans.every(function (t) { return /role="link"/.test(t) && /aria-disabled="true"/.test(t) && !/\bhref=/.test(t); }), 'three inert checkout spans, no href');
assert(!/\bhref="(?!mailto:|tel:|\/supporter-terms\.html|\/privacy\.html|https:\/\/rathor\.ai)/.test(sp), 'supporter section links only to mailto, tel, the terms and privacy pages');
assert(sp.indexOf('Supporter Pass: 1 month') !== -1 && sp.indexOf('Supporter Pass: 3 months') !== -1 && sp.indexOf('Supporter Pass: 12 months') !== -1, 'three passes');
assert(sp.indexOf('US $5, paid once. Does not renew.') !== -1 && sp.indexOf('US $12, paid once. Does not renew.') !== -1 && sp.indexOf('US $35, paid once. Does not renew.') !== -1, 'prices, USD, paid once, does not renew next to each button');
assert((sp.match(/What it is not: no licence, ownership, product, feature, content, support or say in decisions, now or later\. No cash value\./g) || []).length === 4, 'what-it-is-not line on each card and in the disclosure box');
assert(sp.indexOf('Starts the day you pay. Does not renew. You will never be charged again.') !== -1 && sp.indexOf('Full refund for any reason within 30 days') !== -1 && sp.indexOf('18+ only. Canada and US residents only.') !== -1, 'disclosure box (03 A)');
assert(sp.indexOf('Online checkout is not open yet.') !== -1, 'supporter section says checkout is not open');
assert(!/\$\s?(?:50|5\s?(?:USD)?\s?\/\s?month)/.test(sp) && !/\/\s?(?:month|year)\b/.test(sp), 'no recurring price: passes are one-time');
var sterms = read('supporter-terms.html');
var supporters = read('supporters.html');
assert(sterms.indexOf('Version 1.0 · Effective [GO-LIVE MERGE DATE]') !== -1, 'supporter terms date stays a placeholder until go-live');
assert(sterms.indexOf('[DISCORD INVITE LINK]') !== -1, 'Discord invite stays a placeholder until Sherif provides the link');
assert(sterms.indexOf('Supporter Passes are not offered yet.') !== -1 && supporters.indexOf('Supporter Passes are not offered yet.') !== -1, 'supporter pages say passes are not offered yet');
assert(sterms.indexOf('It does not renew. We will never charge you again.') !== -1, 'supporter terms: no renewal');
assert(sterms.indexOf('For 30 days after you buy, you can get a full refund for any reason.') !== -1, 'supporter terms: 30-day refund');
assert(sterms.indexOf('live in Canada or the United States') !== -1 && sterms.indexOf('You must be 18 or older') !== -1, 'supporter terms: Canada + US, 18+');
assert(sterms.indexOf('No GST/HST is charged at checkout at this time. If that changes, GST/HST will be added at checkout.') !== -1, 'supporter terms: neutral GST/HST line');
assert(sterms.indexOf('The Discord role gives no special channels, perks or powers') !== -1, 'supporter terms: role gives no extras');
assert(!/discord\.gg|discord\.com\/invite|discordapp\.com\/invite/i.test(sp + sterms + supporters + priv), 'no real Discord invite link in the supporter pages; the invite stays a placeholder');
assert(!/\b(?:pk|sk|rk|whsec)_[A-Za-z0-9]|\bacct_[A-Za-z0-9]|\bprice_[A-Za-z0-9]{6,}|\bprod_[A-Za-z0-9]{6,}|\bplink_/.test(sp + sterms + supporters), 'no Stripe key or object id on supporter pages');
// Banned wording on every supporter page (Legal Mate 04 section 1).
var banned = /\bdonat|\binvest|\breturns?\b|\bequity\b|\bprofit|priority|early access|coming soon|exclusive|future feature/i;
[['index.html #supporter-pass', sp], ['supporter-terms.html', sterms], ['supporters.html', supporters]].forEach(function (pair) {
  assert(!banned.test(pair[1].replace(/<[^>]+>/g, ' ')), pair[0] + ' uses banned wording: ' + (pair[1].replace(/<[^>]+>/g, ' ').match(banned) || [''])[0]);
});
var supPrivSec = priv.match(/<section id="supporter-pass"[\s\S]*?<\/section>/)[0].replace(/<[^>]+>/g, ' ');
assert(!banned.test(supPrivSec), 'privacy Supporter Pass section uses banned wording');
assert(priv.indexOf('Lattice Chat collect no personal data') === -1, 'privacy must not claim Lattice Chat collects nothing regardless of chosen services');
console.log('licence-self-serve-lock: nothing live (' + all.length + ' files scanned, ' + siteCount + ' site files), two tiers, no auto-renew, privacy truthful');
