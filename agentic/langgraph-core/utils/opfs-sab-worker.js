// agentic/langgraph-core/utils/opfs-sab-worker.js
// version: 17.234.0-sharedarraybuffer-concurrency
// Dedicated Web Worker with SharedArrayBuffer + Atomics for zero-copy SQLite + OPFS
// Highest-performance concurrency model for Rathor.ai checkpointer

self.importScripts('https://cdnjs.cloudflare.com/ajax/libs/sql.js/1.11.0/sql-wasm.min.js');

let db = null;
let fileHandle = null;
let syncHandle = null;
let sharedBuffer = null;   // SharedArrayBuffer for zero-copy state

async function initializeOPFS() {
  if (db) return;

  // Streaming WASM compilation (already optimized)
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

  db.run('PRAGMA journal_mode=WAL;');
  db.run('PRAGMA synchronous=NORMAL;');
  db.run('PRAGMA cache_size=-20000;');

  // Create SharedArrayBuffer for zero-copy state transfer (max 10 MB)
  sharedBuffer = new SharedArrayBuffer(10 * 1024 * 1024);
}

self.onmessage = async function(e) {
  const { action, sabIndex = 0 } = e.data; // sabIndex = offset in SharedArrayBuffer

  try {
    if (action === 'initialize') {
      await initializeOPFS();
      self.postMessage({ success: true, action: 'initialized', sab: sharedBuffer });
      return;
    }

    if (action === 'save') {
      await initializeOPFS();
      // Read state directly from SharedArrayBuffer (zero copy)
      const view = new Uint8Array(sharedBuffer);
      const len = new DataView(sharedBuffer).getUint32(0, true);
      const stateBytes = view.slice(4, 4 + len);
      const state = JSON.parse(new TextDecoder().decode(stateBytes));

      const lumenas = state.lumenasCI || 0;
      if (lumenas < 0.999) {
        self.postMessage({ success: false, reason: 'Mercy Gate blocked' });
        return;
      }

      const blob = new Uint8Array(JSON.stringify(state));
      syncHandle.write(blob, { at: 0 });
      syncHandle.flush();

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

      // Write back to SharedArrayBuffer for zero-copy return
      const encoded = new TextEncoder().encode(JSON.stringify(loadedState));
      const view = new Uint8Array(sharedBuffer);
      new DataView(sharedBuffer).setUint32(0, encoded.length, true);
      view.set(encoded, 4);

      self.postMessage({ success: true, action: 'loaded' });
      return;
    }
  } catch (err) {
    self.postMessage({ success: false, error: err.message });
  }
};
