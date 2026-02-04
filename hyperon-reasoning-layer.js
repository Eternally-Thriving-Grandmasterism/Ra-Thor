// hyperon-reasoning-layer.js – sovereign client-side OpenCog Hypergraph + bidirectional PLN chaining + LQC bounce repulsion
// Forward/backward unification, TV propagation, attention boost, mercy-gated, quantum cosmology symmetry
// MIT License – Autonomicity Games Inc. 2026

let hyperonDB;
const HYPERON_DB_NAME = "rathorHyperonDB";
const HYPERON_STORE = "hyperonHypergraph";

// ────────────────────────────────────────────────────────────────
// Core atom structure (OpenCog Hypergraph style)
class HyperonAtom {
  constructor(handle, type, name = null, tv = { s: 0.5, c: 0.5 }, sti = 0.1, lti = 0.5) {
    this.handle = handle;
    this.type = type;
    this.name = name;
    this.tv = tv;
    this.sti = sti;
    this.lti = lti;
    this.out = [ ];     // incoming
    this.lastUpdate = Date.now();
  }
}

// ────────────────────────────────────────────────────────────────
// Database & seed
async function initHyperonDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(HYPERON_DB_NAME, 1);
    req.onupgradeneeded = evt => {
      const db = evt.target.result;
      if (!db.objectStoreNames.contains(HYPERON_STORE)) {
        const store = db.createObjectStore(HYPERON_STORE, { keyPath: "handle" });
        store.createIndex("type", "type");
        store.createIndex("name", "name");
      }
    };
    req.onsuccess = async evt => {
      hyperonDB = evt.target.result;
      const tx = hyperonDB.transaction(HYPERON_STORE, "readwrite");
      const store = tx.objectStore(HYPERON_STORE);
      const countReq = store.count();
      countReq.onsuccess = async () => {
        if (countReq.result ===
