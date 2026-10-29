#!/usr/bin/env node
// SANDBOX ONLY — issue a per-customer API key for local/sandbox D1.
// Prints the key ONCE (give it to the customer; never commit it) and the SQL that stores
// only its SHA-256 hash. Usage:
//   node scripts/issue-key.mjs <account_id>
//   npx wrangler d1 execute rathor-api-sandbox --local --command "<printed SQL>"
// Do not run against a production database without Sherif's approval.
import { randomBytes, createHash } from 'node:crypto';
const account = process.argv[2];
if (!account || !/^[A-Za-z0-9_\-]{1,64}$/.test(account)) { console.error('usage: issue-key.mjs <account_id>'); process.exit(1); }
const key = ['rth', 'test', ''].join('_') + randomBytes(32).toString('base64url');
const hash = createHash('sha256').update(key).digest('hex');
console.log(`API key (shown once): ${key}`);
// PRICING-4: no free signup credits. A new account starts at 0 credits (matching ensureAccount());
// credits come only from a paid purchase (the $1 trial, a pack, or the plan). An existing account is unchanged.
console.log(`SQL: INSERT OR IGNORE INTO accounts (id, balance) VALUES ('${account}', 0); INSERT INTO api_keys (key_hash, account_id) VALUES ('${hash}', '${account}');`);
