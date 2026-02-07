const DB_NAME = 'RathorNEXiDB';
const DB_VERSION = 6;

const STORES = {
  CHAT_HISTORY: 'chat-history',
  SESSION_METADATA: 'session-metadata',
  TRANSLATION_CACHE: 'translation-cache',
  MERCY_LOGS: 'mercy-logs',
  EVOLUTION_STATES: 'evolution-states',
  USER_PREFERENCES: 'user-preferences'
};

const CACHE_TTL_MS = 7 * 24 * 60 * 60 * 1000;

class RathorIndexedDB {
  constructor() {
    this.db = null;
    this.activeSessionId = localStorage.getItem('rathor_active_session') || 'default';
  }

  async open() {
    if (this.db) return this.db;

    return new Promise((resolve, reject) => {
      const request = indexedDB.open(DB_NAME, DB_VERSION);

      request.onerror = (event) => {
        console.error('[Rathor IndexedDB] Open failed:', event.target.error);
        reject(event.target.error);
      };

      request.onsuccess = (event) => {
        this.db = event.target.result;
        console.log('[Rathor IndexedDB] Opened v' + this.db.version);
        resolve(this.db);
      };

      request.onupgradeneeded = (event) => {
        // ... previous migrations kept unchanged ...
      };

      request.onblocked = () => {
        console.warn('[Rathor IndexedDB] Upgrade blocked — close other tabs');
      };
    });
  }

  // ────────────────────────────────────────────────
  // Translation Metrics with Latency Tracking
  // ────────────────────────────────────────────────

  async getTranslationMetrics() {
    const metrics = await this._transaction(STORES.USER_PREFERENCES, 'readonly', (tx) => {
      const store = tx.objectStore(STORES.USER_PREFERENCES);
      const req = store.get('translation_metrics');
      return new Promise((res, rej) => {
        req.onsuccess = () => res(req.result || {
          total: 0,
          hits: 0,
          misses: 0,
          latencies: [], // array of {ms, timestamp}
          history: []
        });
        req.onerror = () => rej(req.error);
      });
    });

    const hitRate = metrics.total > 0 ? Math.round((metrics.hits / metrics.total) * 100) : 0;

    // Latency stats
    const latencies = metrics.latencies || [];
    const avgLatency = latencies.length > 0 ? Math.round(latencies.reduce((sum, l) => sum + l.ms, 0) / latencies.length) : 0;
    const minLatency = latencies.length > 0 ? Math.min(...latencies.map(l => l.ms)) : 0;
    const maxLatency = latencies.length > 0 ? Math.max(...latencies.map(l => l.ms)) : 0;
    const p95Latency = latencies.length > 0 ? Math.round(latencies.map(l => l.ms).sort((a,b)=>a-b)[Math.floor(latencies.length * 0.95)]) : 0;

    return {
      ...metrics,
      hitRate,
      avgLatency,
      minLatency,
      maxLatency,
      p95Latency,
      cacheSize: await this.getCacheSize()
    };
  }

  async updateTranslationMetrics(isHit, latencyMs = null) {
    const metrics = await this.getTranslationMetrics();
    metrics.total += 1;
    if (isHit) metrics.hits += 1;
    else metrics.misses += 1;

    if (latencyMs !== null && !isHit) {
      metrics.latencies = metrics.latencies || [];
      metrics.latencies.push({ ms: latencyMs, timestamp: Date.now() });
      if (metrics.latencies.length > 1000) metrics.latencies.shift(); // keep last 1000
    }

    metrics.history.push({
      timestamp: Date.now(),
      hit: isHit,
      hitRate: Math.round((metrics.hits / metrics.total) * 100),
      latency: latencyMs
    });
    if (metrics.history.length > 1000) metrics.history.shift();

    await this._transaction(STORES.USER_PREFERENCES, 'readwrite', (tx) => {
      tx.objectStore(STORES.USER_PREFERENCES).put({ key: 'translation_metrics', value: metrics });
    });
  }

  async resetTranslationMetrics() {
    await this._transaction(STORES.USER_PREFERENCES, 'readwrite', (tx) => {
      tx.objectStore(STORES.USER_PREFERENCES).delete('translation_metrics');
    });
  }

  async getCacheSize() {
    return this._transaction(STORES.TRANSLATION_CACHE, 'readonly', (tx) => {
      return tx.objectStore(STORES.TRANSLATION_CACHE).count();
    });
  }

  // ... keep all previous translation cache methods (getCachedTranslation, cacheTranslation, invalidate*, clearExpiredCache) ...
}

export const rathorDB = new RathorIndexedDB();
