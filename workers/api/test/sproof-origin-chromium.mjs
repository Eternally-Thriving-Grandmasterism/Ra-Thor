#!/usr/bin/env node
// SPROOF-FIX-ORIGIN: local headless Chromium proof (not part of `npm test`).
// Serves the sandbox worker on 127.0.0.1, opens the confirm page in Playwright
// Chromium, submits the no-JS form, and records Origin + status. No secrets.
//
// Env vars (both optional):
//   PLAYWRIGHT_MODULE  Module spec or absolute path passed to require() to load
//                      Playwright. Tried first; then '/tmp/sproof-pw/node_modules/playwright',
//                      then plain 'playwright'. No default of its own; if none load, the
//                      script throws "Playwright is not installed".
//   CHROME_PATH        Chromium/Chrome executable passed to chromium.launch({ executablePath }).
//                      Default: /usr/local/bin/google-chrome.
// Example: CHROME_PATH=/usr/bin/chromium PLAYWRIGHT_MODULE=/abs/node_modules/playwright \
//          node test/sproof-origin-chromium.mjs
import http from 'node:http';
import { createRequire } from 'node:module';
import worker from '../src/index.js';
import { AUTH_TOKEN_TTL_SEC } from '../src/config.js';
import { sha256Hex } from '../src/hash.js';
import { randomToken } from '../src/session.js';
import { fakeD1 } from './fakeD1.js';

const require = createRequire(import.meta.url);
function loadPlaywright() {
  const candidates = [
    process.env.PLAYWRIGHT_MODULE,
    '/tmp/sproof-pw/node_modules/playwright',
    'playwright',
  ].filter(Boolean);
  for (const spec of candidates) {
    try { return require(spec); } catch { /* next */ }
  }
  throw new Error('Playwright is not installed. Set PLAYWRIGHT_MODULE or npm install playwright.');
}
const { chromium } = loadPlaywright();

function listen(server) {
  return new Promise((resolve, reject) => {
    server.listen(0, '127.0.0.1', () => resolve(server.address()));
    server.on('error', reject);
  });
}

function close(server) {
  return new Promise((resolve) => server.close(resolve));
}

function nodeHeaders(req) {
  const h = new Headers();
  for (const [k, v] of Object.entries(req.headers)) {
    if (v == null || k === 'transfer-encoding' || k === 'connection') continue;
    if (Array.isArray(v)) for (const item of v) h.append(k, item);
    else h.set(k, v);
  }
  return h;
}

async function readBody(req) {
  const chunks = [];
  for await (const c of req) chunks.push(c);
  return Buffer.concat(chunks);
}

async function main() {
  const sinkHits = [];
  const sink = http.createServer((req, res) => {
    sinkHits.push({
      method: req.method,
      url: req.url,
      origin: req.headers.origin ?? null,
      referer: req.headers.referer ?? null,
    });
    res.writeHead(200, { 'Content-Type': 'text/plain; charset=utf-8', 'Cache-Control': 'no-store' });
    res.end('sproof-sink');
  });
  const sinkAddr = await listen(sink);
  const sinkOrigin = `http://127.0.0.1:${sinkAddr.port}`;

  const env = { ALLOWED_ORIGIN: sinkOrigin, DB: fakeD1() };
  const token = randomToken();
  const now = Math.floor(Date.now() / 1000);
  env.DB.q(
    'INSERT INTO auth_tokens (token_hash, email, expires_at) VALUES (?1, ?2, ?3)',
    await sha256Hex(token),
    'sproof-local@example.com',
    now + AUTH_TOKEN_TTL_SEC,
  );

  const captured = { post: null };
  let apiPort;
  const api = http.createServer(async (req, res) => {
    try {
      const url = `http://127.0.0.1:${apiPort}${req.url}`;
      const method = req.method || 'GET';
      const headers = nodeHeaders(req);
      let body;
      if (method !== 'GET' && method !== 'HEAD') {
        const buf = await readBody(req);
        if (buf.length) body = buf;
      }
      const request = new Request(url, { method, headers, body });
      if (method === 'POST' && new URL(url).pathname === '/v1/auth/verify') {
        captured.post = {
          url,
          origin: request.headers.get('Origin'),
          referer: request.headers.get('Referer'),
        };
      }
      const response = await worker.fetch(request, env);
      if (captured.post && captured.post.status == null && method === 'POST') {
        captured.post.status = response.status;
        captured.post.location = response.headers.get('Location');
      }
      res.statusCode = response.status;
      const cookies = typeof response.headers.getSetCookie === 'function' ? response.headers.getSetCookie() : [];
      response.headers.forEach((value, key) => {
        if (key.toLowerCase() === 'set-cookie') return;
        res.setHeader(key, value);
      });
      for (const c of cookies) res.appendHeader('Set-Cookie', c);
      res.end(Buffer.from(await response.arrayBuffer()));
    } catch (err) {
      res.statusCode = 500;
      res.end(String(err && err.stack ? err.stack : err));
    }
  });
  const apiAddr = await listen(api);
  apiPort = apiAddr.port;
  const apiOrigin = `http://127.0.0.1:${apiPort}`;
  const confirmUrl = `${apiOrigin}/v1/auth/verify?token=${encodeURIComponent(token)}`;

  const browser = await chromium.launch({
    headless: true,
    executablePath: process.env.CHROME_PATH || '/usr/local/bin/google-chrome',
    args: ['--no-sandbox', '--disable-dev-shm-usage'],
  });
  const page = await browser.newPage();
  const getResp = await page.goto(confirmUrl, { waitUntil: 'domcontentloaded' });
  const pageOrigin = new URL(page.url()).origin;
  const action = await page.getAttribute('form', 'action');
  const postTarget = new URL(action || '', page.url());
  const referrerPolicy = getResp.headers()['referrer-policy'];

  const [postResp] = await Promise.all([
    page.waitForResponse((r) => r.request().method() === 'POST' && new URL(r.url()).pathname === '/v1/auth/verify'),
    page.click('button[type=submit]'),
  ]);
  await page.waitForTimeout(250);
  await browser.close();
  await close(api);
  await close(sink);

  const evidence = {
    confirmPageOrigin: pageOrigin,
    postTargetOrigin: postTarget.origin,
    sameOrigin: pageOrigin === postTarget.origin && pageOrigin === apiOrigin,
    referrerPolicy,
    chromiumPostOrigin: captured.post?.origin ?? null,
    chromiumPostStatus: captured.post?.status ?? postResp.status(),
    playwrightPostStatus: postResp.status(),
    postLocation: captured.post?.location ?? null,
    sinkOrigin,
    sinkHits,
    tokenInAnySinkReferer: sinkHits.some((h) => typeof h.referer === 'string' && h.referer.includes(token)),
  };
  const ok = evidence.sameOrigin
    && evidence.referrerPolicy === 'same-origin'
    && evidence.chromiumPostOrigin === apiOrigin
    && evidence.chromiumPostOrigin !== 'null'
    && evidence.chromiumPostStatus === 303
    && evidence.tokenInAnySinkReferer === false;
  process.stdout.write(`${JSON.stringify(evidence, null, 2)}\n`);
  if (!ok) {
    process.stderr.write('sproof-origin-chromium: unexpected result\n');
    process.exit(1);
  }
}

main().catch((err) => {
  process.stderr.write(`${err && err.stack ? err.stack : err}\n`);
  process.exit(1);
});
