// Minimal in-memory D1 stand-in backed by sql.js (SQLite/wasm) for tests.
// Mirrors D1: prepare().bind().first()/run()/all(), and batch() as one transaction.
// changes() inside a batch is the real SQLite function (per previous statement).
// Concurrent batches are serialized (one writer), like SQLite; reads still interleave.
import initSqlJs from 'sql.js';
import { readFileSync } from 'node:fs';
const SQL = await initSqlJs();
const schema = ['0001_init.sql', '0002_unmatched_payments.sql', '0003_trial_redemptions.sql', '0004_auth.sql', '0005_refunds.sql', '0006_early_disputes.sql', '0007_mercy_blocks.sql', '0008_settle_shortfalls.sql', '0009_settle_lock.sql'].map((f) => readFileSync(new URL(`../migrations/${f}`, import.meta.url), 'utf8')).join('\n');
const tick = () => new Promise((r) => setTimeout(r, 0));

export function fakeD1() {
  const db = new SQL.Database();
  db.exec(schema);
  const params = (args) => Object.fromEntries(args.map((v, i) => [`?${i + 1}`, v === undefined ? null : v]));
  const exec = (sql, args) => {
    const st = db.prepare(sql);
    try {
      st.bind(params(args));
      const rows = [];
      while (st.step()) rows.push(st.getAsObject());
      return { results: rows, meta: { changes: db.getRowsModified() } };
    } finally { st.free(); }
  };
  const mk = (sql, args = []) => ({
    sql, args,
    bind: (...a) => mk(sql, a),
    async first() { await tick(); return exec(sql, args).results[0] ?? null; },
    async all() { await tick(); return exec(sql, args); },
    async run() { await tick(); return exec(sql, args); },
  });
  let writeChain = Promise.resolve();
  return {
    raw: db,
    prepare: (sql) => mk(sql),
    async batch(stmts) {
      let release;
      const mine = new Promise((r) => { release = r; });
      const prev = writeChain;
      writeChain = mine;
      await prev;
      await tick();
      try {
        db.exec('BEGIN');
        try {
          const out = stmts.map((s) => exec(s.sql, s.args));
          db.exec('COMMIT');
          return out;
        } catch (e) {
          db.exec('ROLLBACK');
          throw e;
        }
      } finally { release(); }
    },
    q(sql, ...a) { return exec(sql, a).results; },
  };
}
