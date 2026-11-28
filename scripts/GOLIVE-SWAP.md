# Go-live swap (GOLIVE-PREP-1)

Parked prep. Nothing here is live: Stripe stays in test mode, the five buy buttons are inert spans, and no PDF is committed.
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

## Copy flips waiting for Legal Mate or Lead

None of Legal Mate's pack (01 to 04, WILD-COPY-CHANGES-v5) gives verbatim go-live wording for these. Each has `"new": null` in the config, and `--write` stays refused until it is filled. The old text is the exact line in the config.

| Edit | Where |
|---|---|
| 5 button labels | licence.html (2), index.html (3) |
| "Online checkout is not open yet." banner | licence.html |
| "(online checkout is not open yet)" in licenceAskBody | licence.html and 23 packs (English in all) |
| Banner and two meta descriptions | terms-of-sale.html |
| "no instant license key and no public checkout" and "Online checkout is not open yet." | contact.html (and the `contactInquiryLead` key in en.js) |
| #supporter-pass banner | index.html |
| "Supporter Passes are not offered yet." (banner, 2 meta, supporters.html meta and body line) | supporter-terms.html, supporters.html |
| Supporter Pass opener (04 section 4 names it, no wording) | privacy.html, docs/Legal/privacy.md |
| "Saved copy: [... proposed path, Wild to confirm]" | supporter-terms.html |

One flip has verbatim source and is pre-filled: index.html "Saved PDF: https://rathor.ai/[terms/supporter-pass-v1.0.pdf]" loses its brackets (04 section 2 and 03 D give the plain URL). It still needs Sherif's OK of the PDF path (04 section 7).

Not touched by the script: `COMMERCIAL_LICENSE.md` line 82 (already conditional), pilot.html (off limits), and two HTML comments (licence.html "TEST MODE", index.html) that are not visible.
