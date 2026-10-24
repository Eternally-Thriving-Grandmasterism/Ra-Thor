const enc = new TextEncoder();
const hex = (buf) => [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, '0')).join('');

export async function hmacHex(secret, msg) {
  const k = await crypto.subtle.importKey('raw', enc.encode(secret), { name: 'HMAC', hash: 'SHA-256' }, false, ['sign']);
  return hex(await crypto.subtle.sign('HMAC', k, enc.encode(msg)));
}

function safeEq(a, b) {
  if (a.length !== b.length) return false;
  let r = 0;
  for (let i = 0; i < a.length; i++) r |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return r === 0;
}

// Verifies a Stripe-Signature header (t=...,v1=...). Tolerance in seconds.
export async function verifyStripeSignature(payload, header, secret, nowSec = Math.floor(Date.now() / 1000), tolerance = 300) {
  if (!header || !secret) return false;
  const parts = Object.create(null);
  const v1 = [];
  for (const kv of header.split(',')) {
    const [k, v] = kv.split('=');
    if (k === 'v1') v1.push(v); else parts[k] = v;
  }
  const t = parseInt(parts.t, 10);
  if (!t || !v1.length || Math.abs(nowSec - t) > tolerance) return false;
  const expected = await hmacHex(secret, `${t}.${payload}`);
  return v1.some((s) => safeEq(s, expected));
}
