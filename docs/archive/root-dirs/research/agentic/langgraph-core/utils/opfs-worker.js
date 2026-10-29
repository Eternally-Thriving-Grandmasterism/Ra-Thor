// agentic/langgraph-core/utils/opfs-worker.js
// version: 17.233.0-wasm-streaming-compilation
// Dedicated Web Worker for synchronous OPFS + sql.js
// NOW uses WebAssembly.instantiateStreaming for maximum speed
// Fully Mercy-Gated via messages from main thread

self.importScripts('https://cdnjs.cloudflare.com/ajax/libs/sql.js/1.11.0/sql-wasm.min.js');

let db = null;
let fileHandle = null;
let syncHandle = null;

async function initializeOPFS() {
  if (db) return;

  // === STREAMING COMPILATION MAGIC ===
  const response = await fetch('https://cdnjs.cloudflare.com/ajax/libs/sql.js/1.11.0/sql-wasm.wasm');
  const wasm = await WebAssembly.instantiateStreaming(response, {
    // sql.js internal imports (no changes needed)
  });
  const SQL = await sql.default({ wasm });
  // ===================================

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
}

self.onmessage = async function(e) {
  const { action, state, threadId = 'rathor-main-thread' } = e.data;

  try {
    if (action === 'initialize') {
      await initializeOPFS();
      self.postMessage({ success: true, action: 'initialized' });
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
