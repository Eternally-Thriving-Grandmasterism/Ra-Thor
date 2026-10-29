// Launch pricing (PRICING-1). Decided 2026-10-25 by Sherif Botros's councils under his
// delegation, within his 30-60% markup range. Billing stays OFF until RATES_VERIFIED is "true".
// Do not surface these values in public copy until Sherif approves go-live.

// Markup above xAI API list rates: 45% (PRICING-1 council decision, 2026-10-25).
export const MARKUP = 1.45;

// 1 credit = $0.001 (a tenth of a cent). Charges round UP to a whole credit.
export const USD_PER_CREDIT = 0.001;

// xAI API list rates, USD per 1M tokens, from https://docs.x.ai/docs/models (checked 2026-10-25).
// Base rates only; MARKUP is applied in code below. Re-check right before RATES_VERIFIED=true.
export const XAI_RATES_USD_PER_MTOK = Object.freeze({
  'grok-4.7': Object.freeze({ input: 2.0, cachedInput: 0.5, output: 6.0 }),
  'grok-4.3': Object.freeze({ input: 1.25, cachedInput: 0.2, output: 2.5 }),
});

// Credit packs (Stripe TEST mode only). Smallest pack is $10; nothing smaller is offered.
export const MIN_PACK_USD = 10;
export const creditsForUsd = (usd) => Math.round(usd / USD_PER_CREDIT);
export const CREDIT_PACKS = Object.freeze([10, 25, 50].map((usd) => Object.freeze({ usd, credits: creditsForUsd(usd) })));
export const MIN_PACK_CREDITS = creditsForUsd(MIN_PACK_USD);
export const isAllowedPackCredits = (c) => CREDIT_PACKS.some((p) => p.credits === c);

export const RATES_VERIFIED = false; // default; the Worker reads env.RATES_VERIFIED ("true" to enable)
export const PRICING = Object.freeze({
  status: 'LAUNCH_PRICING_RATES_UNVERIFIED',
  usdPerCredit: USD_PER_CREDIT,
  markup: MARKUP,
  monthlyPlanCredits: MIN_PACK_CREDITS, // $10.00 per month
  trialCredits: creditsForUsd(1),       // $1.00 one-time PAID trial (STRIPE_PRICE_TRIAL); no free signup grant
});

export const ratesVerified = (env) => String(env?.RATES_VERIFIED ?? RATES_VERIFIED) === 'true';

// Exact integer math (no float ceil drift): rates as cents per 1M tokens (rate*100) and the
// markup as a percent (MARKUP*100), both integers.
const cents = (usd) => Math.round(usd * 100);
const MARKUP_PCT = Math.round(MARKUP * 100);
const CREDITS_PER_CENT = Math.round(0.01 / USD_PER_CREDIT); // 10
// credits = ceil( sum(tokens * centsPerMTok) * MARKUP_PCT/100 * CREDITS_PER_CENT / 1e6 )
const DENOM = 1e6 * 100;

export function ratesFor(model) {
  const r = XAI_RATES_USD_PER_MTOK[model];
  if (!r) throw new Error(`no rates for model ${model}`);
  return r;
}

export function creditsForTokens(model, tin, tout, cachedIn = 0) {
  const r = ratesFor(model);
  const cached = Math.min(Math.max(0, cachedIn), tin);
  const tokCents = (tin - cached) * cents(r.input) + cached * cents(r.cachedInput) + tout * cents(r.output);
  const num = tokCents * MARKUP_PCT * CREDITS_PER_CENT;
  return Math.max(1, Math.ceil(num / DENOM));
}
// HOLD-HEADROOM-1: reasoning tokens are billed as output. xAI's API reference example reports
// them in usage.completion_tokens_details.reasoning_tokens SEPARATELY from completion_tokens
// (prompt 32 + completion 9 + reasoning 110 = total 151). They are added to output unless
// total_tokens shows they are already inside completion_tokens (total == prompt + completion).
export function billableOutputTokens(usage) {
  const out = usage?.completion_tokens ?? 0;
  const rsn = usage?.completion_tokens_details?.reasoning_tokens;
  if (!Number.isInteger(rsn) || rsn <= 0) return out;
  const p = usage?.prompt_tokens ?? 0; const total = usage?.total_tokens;
  if (Number.isInteger(total) && total === p + out && out >= rsn) return out; // already included
  return out + rsn;
}
export const creditsForUsage = (model, usage) => creditsForTokens(
  model, usage?.prompt_tokens ?? 0, billableOutputTokens(usage), usage?.prompt_tokens_details?.cached_tokens ?? 0);
// MERCY-SAFETY-SCOPE-2: bill a settled call (released, refused by the model, or withheld by
// Mercy-Checks after the call) at the normal rate on the tokens actually used. If upstream
// usage is missing or malformed we cannot know the real cost, so the full hold (the
// worst-case estimate) is charged instead of the 1-credit floor, so it is never below cost.
const validTok = (x) => Number.isInteger(x) && x >= 0;
export function usageIsValid(usage) {
  if (!usage || typeof usage !== 'object' || !validTok(usage.prompt_tokens) || !validTok(usage.completion_tokens)) return false;
  const r = usage.completion_tokens_details?.reasoning_tokens;
  return r === undefined || r === null || validTok(r);
}
export function creditsForSettle(model, usage, hold) {
  return usageIsValid(usage) ? creditsForUsage(model, usage) : hold;
}
export const creditsToUsd = (c) => (c * Math.round(USD_PER_CREDIT * 1e6)) / 1e6;

// Upper bound for a request: every input byte counted as an uncached token (tokens are >= 1
// byte in practice) plus the full capped completion budget.
export function maxCostEstimate(model, promptBytes, maxTokens) {
  return creditsForTokens(model, promptBytes, maxTokens, 0);
}

// HOLD-HEADROOM-1 hold sizing (conservative, named). Prompt bytes already over-count text
// tokens; these add chat-template overhead xAI may count on top of the text.
export const PER_REQUEST_OVERHEAD_TOKENS = 64;
export const PER_MESSAGE_OVERHEAD_TOKENS = 16;
// xAI docs do not say reasoning tokens count against max_tokens, so assume they may exceed
// it. This headroom is held on top of max_tokens; usage above the hold is still billed at
// cost by settleHold (over-hold path), so the headroom only bounds how often that happens.
export const REASONING_HEADROOM_TOKENS = 16384;
export function holdEstimate(model, promptBytes, messageCount, maxTokens) {
  const tin = promptBytes + PER_REQUEST_OVERHEAD_TOKENS + PER_MESSAGE_OVERHEAD_TOKENS * Math.max(0, messageCount);
  return creditsForTokens(model, tin, maxTokens + REASONING_HEADROOM_TOKENS, 0);
}
