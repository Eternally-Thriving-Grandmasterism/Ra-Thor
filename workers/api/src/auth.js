// Per-customer API keys: only SHA-256 hashes are stored (api_keys.key_hash).
// WEB-AUTH-1: authenticate() also accepts a hashed session cookie (src/session.js).
import { accountForKeyHash } from './db.js';
import { sha256Hex, constantTimeEqual } from './hash.js';
import { authenticateSession } from './session.js';

export { sha256Hex, constantTimeEqual };

// Bearer API key (server-side clients). Returns the account id or null.
export async function authenticateBearer(req, env) {
  // Customer keys only: rth_test_ / rth_live_. Stripe sk_/rk_ prefixes never match.
  const m = /^Bearer (rth_(?:test|live)_[A-Za-z0-9_\-]{16,180})$/.exec(req.headers.get('Authorization') || '');
  if (!m) return null;
  const hash = await sha256Hex(m[1]);
  const row = await accountForKeyHash(env.DB, hash);
  // The lookup is by hash already; the constant-time compare is defence in depth (harmless).
  if (!row || !constantTimeEqual(row.key_hash, hash)) return null;
  return row.account_id;
}

// WEB-AUTH-1: Bearer key or a valid session cookie. Both map to the same account ledger.
export async function authenticate(req, env) {
  return (await authenticateBearer(req, env)) || authenticateSession(req, env);
}
