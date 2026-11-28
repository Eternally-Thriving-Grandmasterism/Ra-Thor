# Go-live swap (GOLIVE-PREP-1)

The swap ran on 2026-11-27 (merged ad68cf3c, #62). This file is kept as a record of the procedure; the tokens and old lines below describe the state before the swap.
At go-live one person runs one script, reviews the diff and pushes. The script never calls Stripe and never invents a value, a label or a line of copy.

## The 7 values and where each one lives

Every spot carries a `GOLIVE-SWAP <ID>:` comment next to it. The tokens and the exact old lines are in `scripts/golive-swap.config.json`.

| ID | File | Token |
|---|---|---|
| PAYLINK_100 | licence.html (inert span, `data-checkout-test-link`) | `[TEST_PAYMENT_LINK_100]` |
| PAYLINK_400 | licence.html | `[TEST_PAYMENT_LINK_400]` |
| PAYLINK_SP_1M | index.html, `#supporter-pass` | `[TEST_PAYMENT_LINK_SP_1M]` |
| PAYLINK_SP_3M | index.html | `[TEST_PAYMENT_LINK_SP_3M]` |
| PAYLINK_SP_12M | index.html | `[TEST_PAYMENT_LINK_SP_12M]` |
| DISCORD_INVITE | supporter-terms.html, section 4(b) | `[DISCORD INVITE LINK]` |
| GOLIVE_DATE | terms-of-sale.html ("last updated"), supporter-terms.html ("Effective") | `[GO-LIVE MERGE DATE]` |

The date step also moves the privacy page "Last Updated" date (`2026-10-22`) in privacy.html and the `privacyUpdated` key in all 23 i18n packs (digits only, no translation). Source: PR #60 body, open items.

## The command

Dry run (writes nothing; says READY or NOT READY and lists everything still missing):

    node scripts/golive-swap.js

Swap (clean tree; refuses while anything is missing; `--root` and `--config` allow a rehearsal on a scratch copy):

    node scripts/golive-swap.js --write --date YYYY-MM-DD \
      --discord-invite <https://discord.gg/CODE> \
      --paylink-100 <url> --paylink-400 <url> \
      --paylink-sp-1m <url> --paylink-sp-3m <url> --paylink-sp-12m <url>

Or set `GOLIVE_DATE`, `GOLIVE_DISCORD_INVITE`, `GOLIVE_PAYLINK_100`, `GOLIVE_PAYLINK_400`, `GOLIVE_PAYLINK_SP_1M`, `GOLIVE_PAYLINK_SP_3M`, `GOLIVE_PAYLINK_SP_12M`.
Only the Stripe account owner supplies the five links. A link must be https on the Stripe Payment Link host, a plain `/<id>` path, no userinfo, query, port or test marker. The invite must be `https://discord.gg/<code>` or `https://discord.com/invite/<code>`. The date is `YYYY-MM-DD`.

`--write` then, in order:

1. Turns each inert span into `<a class="rt-btn-ghost" href=... rel="noopener" target="_blank" data-golive-link=ID>LABEL</a>`.
2. Puts the invite and the date in.
3. Applies the copy flips below, using only the `new` text in the config.
4. Builds the frozen PDF `terms/supporter-pass-v1.0.pdf` from the swapped supporter-terms.html (`scripts/build-supporter-terms-pdf.js`, which also runs alone: `node scripts/build-supporter-terms-pdf.js --date YYYY-MM-DD`). It refuses to overwrite an existing frozen PDF. Needs Chrome or Chromium. Output is byte-stable on one machine; commit it once and never edit it. A new version gets a new file name.
5. Patches the tests: the lock test gets the positive pins (below) and the flipped lines; art-uplift-3 (index.html, licence.html) and footer-i18n-1 (23 packs) are re-pinned by sha256.

Then: `node tests/run-node-tests.js` (only the 5 quarantined suites may fail), review `git diff`, `git add` the changed files and the PDF explicitly (never the whitepaper PDF), push, wait for CI. The lock test and the pins are gates: Core votes on them.

## Positive pins the swap installs (not active now)

In `tests/licence-self-serve-lock.test.js`: the five links are listed exactly, per file. Each must parse with `new URL()`, be https on the Stripe Payment Link hostname, have no userinfo, port, query or fragment, a plain alphanumeric path, and no test marker, and appear exactly once as the pinned anchor. Every other buy-like link anywhere still fails the scan, including JS-escaped `https:\/\/` forms (decoded since this PR). The Discord invite is pinned the same way, and the supporter pages may carry no other Discord link than it and Discord's own privacy policy link.

## Copy flips (from Legal Mate WILD-GOLIVE-COPY-v1.md, pasted verbatim)

These are already filled in `scripts/golive-swap.config.json`. `--write` now waits only for the 7 live values (5 Payment Links, Discord invite, date). The PDF path `terms/supporter-pass-v1.0.pdf` is confirmed. privacyRight2 / privacyRight5 stay English (no change).

### Button labels (`config.labels`)

| ID | Old | New |
|---|---|---|
| PAYLINK_100 | Checkout not open yet ($100) | Buy licence, USD $100, no renewal |
| PAYLINK_400 | Checkout not open yet ($400) | Buy licence, USD $400, no renewal |
| PAYLINK_SP_1M | Checkout not open yet (US $5) | Buy 1-month pass, USD $5, no renewal |
| PAYLINK_SP_3M | Checkout not open yet (US $12) | Buy 3-month pass, USD $12, no renewal |
| PAYLINK_SP_12M | Checkout not open yet (US $35) | Buy 12-month pass, USD $35, no renewal |

### Flips (old → new)

- **licence-banner** (licence.html): `<strong class="text-amber-300">Online checkout is not open yet.</strong> Nothing can be bought on this site today. The prices below are what the self-serve licence will cost when checkout opens.` → `<strong class="text-amber-300">Online checkout is open.</strong> You can buy a self-serve 12-month commercial licence below, at the prices shown. Paid once; it does not renew automatically.`
- **licence-ask** / **licence-ask-packs** (licence.html and all 23 packs, English in all): `(online checkout is not open yet)` → `(online checkout is open)`
- **tos-banner** (terms-of-sale.html): `<strong class="text-amber-300">Online checkout is not open yet.</strong> Nothing can be bought on this site today. These terms are published so you can read them before checkout opens.` → `<strong class="text-amber-300">Online checkout is open.</strong> These terms apply when you buy a self-serve 12-month commercial licence on this site.`
- **tos-meta** (terms-of-sale.html, twice, keep the trailing `">`): ` Online checkout is not open yet.">` → ` Online checkout is open.">`
- **contact-nocheckout** / **contact-nocheckout-packs** (contact.html, and the `contactInquiryLead` key in en.js): `There is no instant license key and no public checkout.` → `There is no instant license key. Self-serve checkout is on the licence page.`
- **contact-notopen** (contact.html): ` Online checkout is not open yet. Licence questions:` → ` Self-serve checkout is on the licence page. Licence questions:`
- **index-banner** (index.html `#supporter-pass`): `<strong class="text-amber-300">Online checkout is not open yet.</strong> Nothing can be bought on this site today.` → `<strong class="text-amber-300">Online checkout is open.</strong> You can buy a one-time Supporter Pass below.`
- **sterms-banner** (supporter-terms.html): `<strong class="text-amber-300">Supporter Passes are not offered yet.</strong> Nothing can be bought on this site today. These terms are published so you can read them before checkout opens.` → `<strong class="text-amber-300">Supporter Passes are available to buy.</strong> These terms apply when you buy a one-time Supporter Pass on this site.`
- **sterms-meta** (supporter-terms.html, twice) and **supporters-meta** (supporters.html, once): ` Supporter Passes are not offered yet.">` → ` Supporter Passes are available to buy.">`
- **supporters-body** (supporters.html): `No names are listed yet. Supporter Passes are not offered yet.` → `No names are listed yet. You can buy a one-time Supporter Pass on the home page.`
- **privacy-sp-html** / **privacy-sp-md** (privacy.html and docs/Legal/privacy.md): `Supporter Passes are not offered yet. This section applies only once they are, and only if you buy one.` → `Supporter Passes are available to buy. This section applies only if you buy one.`
- **saved-copy-terms** (supporter-terms.html): `Saved copy: [https://rathor.ai/terms/supporter-pass-v1.0.pdf — proposed path, Wild to confirm]` → `Saved copy: https://rathor.ai/terms/supporter-pass-v1.0.pdf`
- **saved-pdf-index** (index.html, already pre-filled): `Saved PDF: https://rathor.ai/[terms/supporter-pass-v1.0.pdf]` → `Saved PDF: https://rathor.ai/terms/supporter-pass-v1.0.pdf`

Not touched by the script: `COMMERCIAL_LICENSE.md` line 82 (already conditional), pilot.html (off limits), two HTML comments (licence.html "TEST MODE", index.html) that are not visible, and privacyRight2 / privacyRight5 (stay English).
