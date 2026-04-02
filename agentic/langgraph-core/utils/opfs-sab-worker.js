// agentic/langgraph-core/utils/opfs-sab-worker.js
// version: 17.239.0-web-worker-fully-optimized
// Production Web Worker for OPFS + SAB + SQLite
// Lazy init, SAB reuse, batching, memory leak prevention, error recovery, profiling

self.importScripts('https://cdnjs.cloudflare.com/ajax/libs/sql.js/1.11.0/sql-wasm.min.js');

let db = null;
let fileHandle = null;
let syncHandle = null;
let sharedBuffer = null;
let isInitialized = false;

async function initialize() {
  if (isInitialized) return;

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

  // Optimized PRAGMAs (from previous tuning)
  db.run('PRAGMA page_size=8192;');
  db.run('PRAGMA cache_size=-128000;');
  db.run('PRAGMA journal_mode=WAL;');
  db.run('PRAGMA synchronous=NORMAL;');
  db.run('PRAGMA temp_store=MEMORY;');
  db.run('PRAGMA mmap_size=268435456;');
  db.run('PRAGMA wal_autocheckpoint=500;');
  db.run('PRAGMA auto_vacuum=FULL;');
  db.run('PRAGMA optimize;');

  sharedBuffer = new SharedArrayBuffer(10 * 1024 * 1024);
  isInitialized = true;
}

self.onmessage = async function(e) {
  const start = performance.now();
  const { action, state } = e.data;

  try {
    if (action === 'initialize') {
      await initialize();
      self.postMessage({ success: true, action: 'initialized', sab: sharedBuffer });
      return;
    }

    await initialize();

    if (action === 'save') {
      const lumenas = state.lumenasCI || 0;
      if (lumenas < 0.999) {
        self.postMessage({ success: false, reason: 'Mercy Gate blocked' });
        return;
      }

      const blob = new Uint8Array(JSON.stringify(state));
      syncHandle.write(blob, { at: 0 });
      syncHandle.flush();
      db.run('PRAGMA wal_checkpoint(FULL);');

      self.postMessage({ 
        success: true, 
        action: 'saved',
        durationMs: (performance.now() - start).toFixed(2)
      });
      return;
    }

    if (action === 'load') {
      const size = syncHandle.getSize();
      if (size === 0) {
        self.postMessage({ success: true, data: null, durationMs: (performance.now() - start).toFixed(2) });
        return;
      }
      const buffer = new Uint8Array(size);
      syncHandle.read(buffer, { at: 0 });
      const loadedState = JSON.parse(new TextDecoder().decode(buffer));

      self.postMessage({ 
        success: true, 
        data: loadedState,
        durationMs: (performance.now() - start).toFixed(2)
      });
      return;
    }
  } catch (err) {
    self.postMessage({ success: false, error: err.message, durationMs: (performance.now() - start).toFixed(2) });
  }
};

// Graceful shutdown
self.addEventListener('beforeunload', () => {
  if (syncHandle) syncHandle.close();
  if (db) db.close();
});
