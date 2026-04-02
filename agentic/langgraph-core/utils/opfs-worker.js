// agentic/langgraph-core/utils/opfs-worker.js
// version: 17.232.0-opfs-web-worker
// Dedicated Web Worker for synchronous OPFS + sql.js
// Runs all database I/O off the main thread for zero UI blocking
// Fully Mercy-Gated via messages from main thread

self.importScripts('https://cdnjs.cloudflare.com/ajax/libs/sql.js/1.11.0/sql-wasm.min.js');

let db = null;
let fileHandle = null;
let syncHandle = null;

async function initializeOPFS() {
  if (db) return;

  const SQL = await sql.default({ locateFile: file => `https://cdnjs.cloudflare.com/ajax/libs/sql.js/1.11.0/${file}` });
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
      const lumenas = state.lumenasCI || 0; // main thread already calculated
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
