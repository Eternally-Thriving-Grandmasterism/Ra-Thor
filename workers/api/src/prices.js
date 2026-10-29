// PRICING-2: server-side table of what each Stripe Price grants. Credits are NEVER taken
// from client-supplied metadata. Price ids come from env vars (wrangler.toml [vars]); they
// are not secrets. Amounts are in US cents and are cross-checked against what Stripe charged.
import { CREDIT_PACKS, PRICING } from './credits.js';

export const CURRENCY = 'usd';
const cents = (usd) => Math.round(usd * 100);

// kind: 'pack' (one-time, checkout mode=payment), 'trial' (one-time, mode=payment),
//       'plan' (subscription: checkout mode=subscription for month one, invoice.paid
//       billing_reason=subscription_cycle for renewals).
export const PRICE_ENV = Object.freeze([
  ...CREDIT_PACKS.map((p) => Object.freeze({ env: `STRIPE_PRICE_PACK_${p.usd}`, kind: 'pack', amount: cents(p.usd), credits: p.credits })),
  Object.freeze({ env: 'STRIPE_PRICE_PLAN_MONTHLY', kind: 'plan', amount: cents(10), credits: PRICING.monthlyPlanCredits }),
  Object.freeze({ env: 'STRIPE_PRICE_TRIAL', kind: 'trial', amount: cents(1), credits: PRICING.trialCredits }),
]);

export function priceTable(env) {
  const m = new Map();
  for (const p of PRICE_ENV) {
    const id = typeof env?.[p.env] === 'string' ? env[p.env].trim() : '';
    if (id) m.set(id, p);
  }
  return m;
}

// Allowed kinds per payment context. A pack price can never satisfy a plan context and vice versa.
// WEB-AUTH-1: client sends one of these keys, never a Stripe price id or amount.
export const CHECKOUT_PRICE_KEYS = Object.freeze({
  pack_10: Object.freeze({ env: 'STRIPE_PRICE_PACK_10', mode: 'payment' }),
  pack_25: Object.freeze({ env: 'STRIPE_PRICE_PACK_25', mode: 'payment' }),
  pack_50: Object.freeze({ env: 'STRIPE_PRICE_PACK_50', mode: 'payment' }),
  plan_monthly: Object.freeze({ env: 'STRIPE_PRICE_PLAN_MONTHLY', mode: 'subscription' }),
  trial: Object.freeze({ env: 'STRIPE_PRICE_TRIAL', mode: 'payment', trial: true }),
});

export function resolveCheckoutPrice(env, key) {
  const spec = CHECKOUT_PRICE_KEYS[key];
  if (!spec) return { reason: 'unknown_price_key' };
  const id = typeof env?.[spec.env] === 'string' ? env[spec.env].trim() : '';
  if (!id) return { reason: 'price_not_configured', env: spec.env };
  return { spec, priceId: id };
}

export const CONTEXT_KINDS = Object.freeze({
  checkout_payment: ['pack', 'trial'],
  checkout_subscription: ['plan'],
  invoice_cycle: ['plan'],
});

// PRICING-3: resolve ONE Stripe line item to a server-side entry or a refusal reason.
// Grants happen only by a configured price id (never by amount), with quantity 1, currency
// usd, and the Price's own unit_amount equal to the table amount. Totals (amount_total /
// amount_paid) are NOT compared: Stripe Tax adds to them and promo codes subtract from them.
// When the line's pre-tax, pre-discount amount_subtotal is present it must also equal the table
// amount. A discounted purchase grants the credits of the price bought: promos discount money,
// not credits.
//   item: { priceId, unitAmount, priceCurrency, quantity, subtotal, currency }
export function resolveLineItem(env, context, item) {
  const kinds = CONTEXT_KINDS[context];
  if (!kinds) return { reason: `unsupported_context_${context}` };
  if (!item) return { reason: 'no_line_items' };
  if (item.priceId === 'multiple_line_items') return { reason: 'multiple_line_items' };
  if (!item.priceId) return { reason: 'price_missing' };
  const lower = (c) => (typeof c === 'string' ? c.toLowerCase() : c);
  for (const c of [item.priceCurrency, item.currency]) if (c != null && lower(c) !== CURRENCY) return { reason: 'currency_mismatch' };
  if (item.priceCurrency == null && item.currency == null) return { reason: 'currency_missing' };
  const p = priceTable(env).get(item.priceId);
  if (!p) return { reason: 'unknown_price' };
  if (!kinds.includes(p.kind)) return { reason: `price_kind_${p.kind}_not_allowed_for_${context}` };
  if (item.quantity !== 1) return { reason: 'quantity_not_1' };
  if (item.unitAmount != null && item.unitAmount !== p.amount) return { reason: 'amount_mismatch' };
  if (item.subtotal != null && item.subtotal !== p.amount) return { reason: 'amount_mismatch' };
  if (item.unitAmount == null && item.subtotal == null) return { reason: 'amount_missing' };
  return { entry: p, matchedBy: 'price_id' };
}

// Checkout Session line items as returned by GET /v1/checkout/sessions/{id}/line_items.
// Returns null (no items), or a normalised item ('multiple_line_items' if more than one).
export function checkoutLineItem(list) {
  const li = list?.data;
  if (!Array.isArray(li) || li.length === 0) return null;
  if (li.length > 1 || list.has_more) return { priceId: 'multiple_line_items' };
  const x = li[0];
  return { priceId: x?.price?.id ?? null, unitAmount: x?.price?.unit_amount ?? null, priceCurrency: x?.price?.currency ?? null, quantity: x?.quantity ?? null, subtotal: Number.isInteger(x?.amount_subtotal) ? x.amount_subtotal : null, currency: x?.currency ?? null };
}

// Invoice line from invoice.paid (price id from the line itself, never metadata).
// Older API: lines.data[0].price.{id,unit_amount}; newer: .pricing.price_details.price + .pricing.unit_amount_decimal.
export function invoiceLineItem(o) {
  const l = o?.lines?.data;
  if (!Array.isArray(l) || l.length === 0) return null;
  if (l.length > 1) return { priceId: 'multiple_line_items' };
  const x = l[0];
  const priceId = x?.price?.id ?? (typeof x?.price === 'string' ? x.price : null) ?? x?.pricing?.price_details?.price ?? null;
  let unitAmount = x?.price?.unit_amount ?? null;
  if (unitAmount == null && x?.pricing?.unit_amount_decimal != null) { const n = Number(x.pricing.unit_amount_decimal); unitAmount = Number.isInteger(n) ? n : NaN; }
  return { priceId, unitAmount, priceCurrency: x?.price?.currency ?? null, quantity: x?.quantity ?? null, subtotal: null, currency: x?.currency ?? o?.currency ?? null };
}
