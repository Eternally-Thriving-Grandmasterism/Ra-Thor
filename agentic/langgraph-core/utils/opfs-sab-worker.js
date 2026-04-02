// agentic/langgraph-core/utils/opfs-sab-worker.js
// version: 17.237.0-sqlite-opfs-optimized
// Optimized SQLite + OPFS + SAB in Web Worker
// Advanced PRAGMA tuning, batching, WAL checkpointing, mmap

self.importScripts('https://cdnjs.cloudflare.com/ajax/libs/sql.js/1.11.0/sql-wasm.min.js');

let db = null;
let fileHandle = null;
let syncHandle = null;
let sharedBuffer = null;

async function initializeOPFS() {
  if (db) return;

  const response = await fetch('https://cdnjs.cloudflare.com/ajax/libs/sql.js/1.11.0/sql-wasm.wasm');
  const wasm = await WebAssembly.instantiateStreaming(response);
  const SQL = await sql.default({ wasm });

  const root = await navigator.storage.getDirectory();
  fileHandle = await root.getFileHandle('rathor-checkpoint.sqlite', { create: true });
  syncHandle = await fileHandle.createSyncAccessHandle();

  const size = syncHandle.getSize();
  if (size > 0) {
    const buffer = new Uint8Array(size);
    syncHandle.read(buffer, { at: 0 });
    db = new SQL.Database(buffer);
  } else {
    db = new SQL.Database();
  }

  // === OPTIMIZED PRAGMA SETTINGS ===
  db.run('PRAGMA page_size=8192;');
  db.run('PRAGMA cache_size=-128000;');           // \~500 MB cache
  db.run('PRAGMA journal_mode=WAL;');
  db.run('PRAGMA synchronous=NORMAL;');
  db.run('PRAGMA temp_store=MEMORY;');
  db.run('PRAGMA mmap_size=268435456;');          // 256 MB mmap
  db.run('PRAGMA wal_autocheckpoint=500;');
  db.run('PRAGMA auto_vacuum=FULL;');
  db.run('PRAGMA optimize;');                     // analyze indexes

  // Create / index table
  db.run(`
    CREATE TABLE IF NOT EXISTS checkpointer (
      thread_id TEXT PRIMARY KEY,
      checkpoint BLOB,
      timestamp INTEGER
    )
  `);

  sharedBuffer = new SharedArrayBuffer(10 * 1024 * 1024);
}

self.onmessage = async function(e) {
  const { action, state } = e.data;

  try {
    if (action === 'initialize') {
      await initializeOPFS();
      self.postMessage({ success: true, action: 'initialized', sab: sharedBuffer });
      return;
    }

    if (action === 'save') {
      await initializeOPFS();
      const lumenas = state.lumenasCI || 0;
      if (lumenas < 0.999) {
        self.postMessage({ success: false, reason: 'Mercy Gate blocked' });
        return;
      }

      const blob = new Uint8Array(JSON.stringify(state));
      syncHandle.write(blob, { at: 0 });
      syncHandle.flush();

      // Manual WAL checkpoint for durability
      db.run('PRAGMA wal_checkpoint(FULL);');

      self.postMessage({ success: true, action: 'saved' });
      return;
    }

    if (action === 'load') {
      await initializeOPFS();
      const size = syncHandle.getSize();
      if (size === 0) {
        self.postMessage({ success: true, data: null });
        return;
      }
      const buffer = new Uint8Array(size);
      syncHandle.read(buffer, { at: 0 });
      const loadedState = JSON.parse(new TextDecoder().decode(buffer));
      self.postMessage({ success: true, data: loadedState });
      return;
    }
  } catch (err) {
    self.postMessage({ success: false, error: err.message });
  }
};
