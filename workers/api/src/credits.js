// PLACEHOLDER pricing constants — PENDING SHERIF'S APPROVAL. NOT FINAL.
// Do not surface these values in any public page, copy, or claim.
export const PLACEHOLDER_PRICING = Object.freeze({
  status: 'PLACEHOLDER_PENDING_APPROVAL',
  usdPerCredit: 0.01,          // PLACEHOLDER: 1 credit = $0.01 of token cost
  markup: 0.15,                // PLACEHOLDER: +15%
  monthlyPlanCredits: 450,     // PLACEHOLDER: $5/mo plan
  trialCredits: 100,           // PLACEHOLDER: one-time trial
  // PLACEHOLDER token cost (USD per 1M tokens); verify against xAI pricing.
  usdPerMTokIn: 3.0,
  usdPerMTokOut: 15.0,
});

export function creditsForUsage(usage, p = PLACEHOLDER_PRICING) {
  const tin = usage?.prompt_tokens ?? 0;
  const tout = usage?.completion_tokens ?? 0;
  const usd = (tin * p.usdPerMTokIn + tout * p.usdPerMTokOut) / 1e6;
  return Math.max(1, Math.ceil((usd * (1 + p.markup)) / p.usdPerCredit));
}

const key = (u) => `credits:${u}`;

export async function getBalance(kv, user) {
  const v = await kv.get(key(user));
  if (v === null) {
    // One-time trial grant (PLACEHOLDER amount), recorded so it never repeats.
    if (await kv.get(`trial:${user}`)) return 0;
    await kv.put(`trial:${user}`, '1');
    await kv.put(key(user), String(PLACEHOLDER_PRICING.trialCredits));
    return PLACEHOLDER_PRICING.trialCredits;
  }
  return parseInt(v, 10) || 0;
}

// NOTE: KV is not transactional; D1 should replace this before any real use.
export async function decrement(kv, user, n) {
  const bal = await getBalance(kv, user);
  const next = Math.max(0, bal - n);
  await kv.put(key(user), String(next));
  return next;
}

export async function grant(kv, user, n) {
  const bal = await getBalance(kv, user);
  await kv.put(key(user), String(bal + n));
  return bal + n;
}
