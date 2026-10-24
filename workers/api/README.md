# Ra-Thor API Worker (api.rathor.ai) — SANDBOX DRAFT

Status: **not deployed**. No Cloudflare, Stripe, or Resend resources have been created. Stripe handling is **test mode only** (events with `livemode: true` are refused).

Ported from `app/api/grok/route.js`. Seller of hosted credits: PENDING Sherif (proposed: Autonomicity Games Inc.).

## Routes
- `GET /health`
- `POST /v1/chat` — proxies to the xAI Grok API (`XAI_API_KEY` secret); per-user credit check/decrement in KV; CORS limited to `https://rathor.ai`.
- `POST /v1/stripe/webhook` — verifies `Stripe-Signature` with `STRIPE_WEBHOOK_SECRET`; on `checkout.session.completed` grants credits (idempotent per event id) and sends a receipt via Resend from `Ra-Thor <info@rathor.ai>` (`RESEND_API_KEY`).

## Pricing
All values in `src/credits.js` are **PLACEHOLDER, pending Sherif's approval**. They are not final and must not appear in public copy.

## Open items before any real use
- User auth is a placeholder (`token:<t>` → user id in KV).
- KV is not atomic; move balances to D1 before real traffic.
- Streaming is disabled so usage can be metered.
- Mercy-Checks integration not yet wired.

## Test
```
npm install && npm test
```
