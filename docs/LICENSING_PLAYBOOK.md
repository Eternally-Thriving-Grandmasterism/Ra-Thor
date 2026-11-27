# Licensing Playbook: selling a commercial software licence online

**Status: DRAFT, internal, for maintainers only. Not legal or tax advice.**

This is a reusable, product-agnostic checklist for selling a fixed-term commercial licence through a one-time Stripe payment. It was written from the perspective of an Ontario, Canada seller. It makes no public claims. Any customer-facing wording below is a *template* that the owner must approve before it is published.

The research was done on 2026-11-26. Re-check every source before each launch, because laws and Stripe's terms change.

Defaults when unsure:
- **UNCONFIRMED** means the point could not be confirmed from a primary source.
- The house rule for anything UNCONFIRMED is to take the **minimal or zero-risk option** until counsel or an accountant says otherwise.
- Items marked ⚖️ need a lawyer. Items marked 🧾 need an accountant.

---

## 0. Pre-flight (owner decisions)

- [ ] Product name, tiers, price **in a stated currency** (e.g. "USD 100 / 12 months"), term, and the "larger use: contact us for a quote" path.
- [ ] One-time payment with **no auto-renew**. This avoids the subscription and renewal rules, which are not covered here.
- [ ] Seller identity to be shown: legal name or business name, postal address, email, phone.
- [ ] Countries you will sell to. Restricting countries is the zero-risk default.
- [ ] Owner sign-off that the site's privacy claims will change before checkout goes live (see §6).

## 1. B2B-only gating

**Why:** Ontario's Consumer Protection Act, 2002 defines a "consumer" as "an individual acting for personal, family or household purposes and does not include a person who is acting for business purposes" (s.1). Its internet-agreement rules (ss.37–40) protect consumers only. https://www.ontario.ca/laws/statute/02c30

Checklist:
- [ ] Label the offer as a **commercial / business licence**.
- [ ] At checkout, require a **business name**. Stripe Checkout can collect the name and an optional tax ID: https://docs.stripe.com/tax/checkout/tax-ids
- [ ] Add a required checkbox: "I am purchasing on behalf of a business, for business purposes, and not as a consumer."
- [ ] Refund or decline orders that are clearly personal.

**UNCONFIRMED** ⚖️: a self-declaration may not be enough. The CPA looks at what the buyer was actually doing (s.1), it bars avoidance (s.3), and consumer rights can't be waived (s.7). **Zero-risk default: also meet the internet-agreement steps in §2,** which cost nothing.

CPA 2023: passed (S.O. 2023 c.23, Sched. 1) but comes into force by proclamation. As of 2026-11-26 it appears **not in force (UNCONFIRMED)**. Re-check https://www.ontario.ca/laws/statute/s23023 before each launch.

## 2. Clickwrap acceptance + the "written and signed" problem

**Law:** Ontario Electronic Commerce Act, 2000: https://www.ontario.ca/laws/statute/00e17
- s.19(1): an offer or acceptance can be expressed by "clicking on an appropriate icon", "unless the parties agree otherwise" (s.19(2)).
- s.11(1): an electronic signature satisfies a **legal** requirement that a document be signed.

**Problem pattern:** a project licence that says a commercial licence must be "written and signed" sets a *contractual* condition, not a legal requirement. Relying on s.11 is therefore UNCONFIRMED, and s.19(2) lets the licence override clickwrap.

**Fix (template; final wording ⚖️):** amend the licence clause to allow electronic formation. For example:

> "A Commercial Licence may be entered into electronically, including by the licensee's electronic acceptance of the Commercial Licence Terms at checkout together with payment. The parties agree that such acceptance and the licensor's emailed confirmation together constitute a written agreement signed by both parties for the purposes of this section."

Checkout flow, which also meets CPA 2002 ss.38–39 and O. Reg. 17/05 ss.31–33 (https://www.ontario.ca/laws/regulation/050017):
- [ ] **Before payment**, show the full Commercial Licence Terms in a form the buyer can save and print. Include:
  - seller name, address, phone and email;
  - description of the licence;
  - itemised price and currency;
  - taxes;
  - term (12 months, no auto-renew);
  - refund and cancellation policy;
  - restrictions.
- [ ] An **unticked "I accept the Commercial Licence Terms" checkbox** plus a clear Pay button. Allow the buyer to review and correct before paying (s.38(2)).
- [ ] Within **15 days**, deliver a written copy (Reg. s.33(1)). The confirmation email should include the terms (or a versioned PDF), licensee name, tier, start and end dates, and amount.
- [ ] Keep a record of: terms version, acceptance timestamp, buyer business name and email, Stripe payment ID.
- If these steps are missed with a consumer: cancellation is possible within 7 days of receiving the copy, or within 30 days if no copy is sent (s.40).

## 3. Tax setup 🧾

### Canada (GST/HST)
- [ ] **Small supplier:** no registration is required while taxable revenue is **≤ CAD 30,000 over four consecutive calendar quarters**. If you exceed it within **one** quarter, tax applies from the sale that crosses it. Revenue counts **worldwide, including zero-rated sales**. https://www.canada.ca/en/revenue-agency/services/tax/businesses/topics/gst-hst-businesses/when-register-charge.html
- [ ] Track cumulative revenue **in CAD** each quarter, including USD sales converted.
- [ ] **Non-resident, unregistered buyers:** a licence of intangible personal property is **zero-rated** (ETA Sched. VI Part V s.10.1, and s.10 for IP licences). Exclusions include individuals who are in Canada at the time, and property usable only in Canada. https://laws-lois.justice.gc.ca/eng/acts/E-15/section-sched201390.html/FullText.html. Keep residency evidence (billing country, business address): CRA Memo 4.5.3, https://www.canada.ca/en/revenue-agency/services/forms-publications/publications/4-5-3/exports-services-intangible-personal-property.html
- [ ] **Once registered:** charge GST/HST for the buyer's province (rates: https://www.canada.ca/en/revenue-agency/services/tax/businesses/topics/gst-hst-businesses/charge-collect-which-rate/calculator.html). **UNCONFIRMED** 🧾: place-of-supply details, and whether PST or QST applies.
- Zero-risk default: stay an unregistered small supplier and register *before* the threshold on an accountant's advice.

### United States
- [ ] States may require remote (including foreign) sellers to collect sales tax once economic-nexus thresholds are crossed: *South Dakota v. Wayfair* (2018), https://www.supremecourt.gov/opinions/17pdf/585us1r58_pok0.pdf
- **UNCONFIRMED** 🧾: per-state thresholds (commonly about USD 100k and/or 200 transactions), how each state applies them to foreign sellers, and whether software or SaaS is taxable there. Monitor sales per state.

### Stripe Tax
- [ ] Turn on **threshold monitoring**. It is listed as included in Tax Basic: https://stripe.com/tax/pricing
- Calculation fees apply only "where you're registered to collect taxes": 0.5% of volume on Checkout and Payment Links (percentage from Stripe pricing via a secondary source; confirm in the Dashboard). Without registrations, Stripe returns zero tax: https://docs.stripe.com/tax/payment-links
- **UNCONFIRMED:** confirm that monitoring alone incurs **$0**. Tax Complete (registrations and filings) is a paid plan. Do not enable it without owner approval.

## 4. Stripe eligibility and site disclosures

- [ ] Check the product against Stripe's Prohibited and Restricted Businesses list: https://stripe.com/legal/restricted-businesses
  - Sanctioned jurisdictions are blocked, and there are Russia-specific software and IT bans. Restrict those countries in Checkout.
  - Selling software "without appropriate authorization" is prohibited. Confirm you own or have rights to everything licensed ⚖️.
  - "Exclusive content or digital goods" is a *restricted* category, so expect possible extra review (**UNCONFIRMED** whether it applies).
- [ ] Stripe website checklist: https://docs.stripe.com/get-started/checklist/website
  - [ ] Clear description of what is sold.
  - [ ] Currency stated explicitly (e.g. "USD", not just "$").
  - [ ] Customer-service contact: email **and** another direct channel (e.g. phone), not only a form.
  - [ ] Refund policy, and a cancellation policy if applicable.
  - [ ] Legal or export restrictions.
  - [ ] Privacy policy.
  - [ ] Site public (no password) and matching the business name given to Stripe (per Stripe support FAQ, quoted by secondary sources; **UNCONFIRMED** wording).
- [ ] Stay in **test mode** until the owner approves going live.

## 5. Refund and contact pages (templates; owner approval required ⚖️)

**Contact page:** seller legal or business name, postal address, support email, phone, expected response time, and an "Enterprise / larger use: request a quote" contact.

**Refund policy (example structure only; pick terms deliberately):**
- Licences are one-time purchases for a fixed 12-month term and **do not auto-renew**.
- Refund window: e.g. "full refund on request within N days of purchase if the licence key or download has not been used", or another rule. *The value is the owner's decision.*
- Duplicate or erroneous charges are refunded.
- How to request: email the support address with the order ID.
- Statutory rights: "Nothing in this policy limits any rights you have under applicable law."

## 6. Privacy wording once checkout collects data

- [ ] **Retire any "we collect no personal data" claim** (and update any tests that lock it) **before** checkout goes live. A paid order collects at least name, email, billing country or address, business name, and possibly a tax ID.
- Law: PIPEDA's 10 fair information principles: https://www.priv.gc.ca/en/privacy-topics/privacy-laws-in-canada/the-personal-information-protection-and-electronic-documents-act-pipeda/p_principle/
- Third-party and cross-border processing: OPC guidelines, https://www.priv.gc.ca/en/privacy-topics/airports-and-borders/gl%5Fdab%5F090127/, and the online-consent FAQ, https://www.priv.gc.ca/en/privacy-topics/privacy-for-businesses/appropriate-handling-of-personal-information/collecting-personal-information-and-consent/consent/gl_oc_201405_faq/

Template sections (final wording ⚖️):
1. **Who is responsible:** [name], contact [email] (accountability).
2. **What we collect at purchase:** name, email, business name, billing address or country, tax ID (if given), and payment details processed by Stripe. Card numbers are handled by Stripe, not by us.
3. **Why:** issue and administer the licence, send receipts, meet tax and accounting obligations, provide support, prevent fraud. We do not send marketing without separate opt-in consent.
4. **Processors and location:** "Payments are processed by Stripe. Your information may be stored or processed outside Canada (for example in the United States) and may be accessible to courts, law enforcement and national security authorities there." Link Stripe's privacy policy.
5. **Retention:** kept as long as needed for the purposes above and as required by tax law. **UNCONFIRMED** 🧾: confirm the CRA record-keeping period, generally 6 years.
6. **Safeguards.**
7. **Your rights:** access, correction, how to complain, including to the Office of the Privacy Commissioner of Canada.
8. **Rest of site:** say what the rest of the site does or doesn't collect, only if it is true and verified.

**UNCONFIRMED** ⚖️: GDPR or UK GDPR applies if you sell to the EU or UK. Zero-risk default: restrict checkout countries until advised.

## 7. Lawyer-only / accountant-only items

⚖️ Lawyer:
1. Final wording of the licence-clause amendment allowing electronic formation, and its effect on existing users or licensees.
2. Commercial Licence Terms: scope of use, warranty disclaimer, limitation of liability, indemnity, governing law and forum, termination.
3. IP ownership chain: third-party, open-source and AI-generated components, and licence compatibility.
4. Whether B2B gating removes CPA obligations, and how to handle consumers who get through anyway.
5. Refund and cancellation policy wording.
6. Privacy policy wording, and foreign privacy law (GDPR and others) if selling abroad.
7. Export controls and sanctions screening for the software (Canada Export Control List, US EAR). **Not researched: UNCONFIRMED.**
8. Business structure and business-name registration. **Not researched: UNCONFIRMED.**

🧾 Accountant:
1. When to register for GST/HST.
2. Place-of-supply rules and PST/QST.
3. Threshold tracking in CAD.
4. US state nexus.
5. Income-tax reporting.
6. Record-retention period.

## 8. Launch gate (all must be true)

- [ ] §0 decisions made by the owner.
- [ ] Licence clause amended (lawyer-reviewed) and merged.
- [ ] Terms, refund, contact and privacy pages approved by the owner (and a lawyer where marked) and published.
- [ ] Checkout includes the B2B checkbox, the accept-terms checkbox, explicit currency, country restrictions, and a confirmation email containing the terms.
- [ ] Stripe Tax monitoring on; revenue tracker in CAD.
- [ ] Every UNCONFIRMED item above is resolved or explicitly accepted by the owner.
- [ ] Owner approves switching Stripe from test mode to live.

*Source research: `research-2026-11-26.md` (internal).*
