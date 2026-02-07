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
  // Bulk Delete Methods (new)
  // ────────────────────────────────────────────────

  /**
   * Delete all messages belonging to a specific session
   * @param {string} sessionId
   * @returns {Promise<number>} Number of deleted records
   */
  async bulkDeleteMessagesBySession(sessionId) {
    return this._transaction(STORES.CHAT_HISTORY, 'readwrite', (tx) => {
      const store = tx.objectStore(STORES.CHAT_HISTORY);
      const index = store.index('sessionId');
      const request = index.openCursor(IDBKeyRange.only(sessionId));
      let count = 0;

      return new Promise((resolve, reject) => {
        request.onsuccess = (event) => {
          const cursor = event.target.result;
          if (cursor) {
            cursor.delete();
            count++;
            cursor.continue();
          } else {
            resolve(count);
          }
        };
        request.onerror = () => reject(request.error);
      });
    });
  }

  /**
   * Bulk invalidate (delete) all translation cache entries for a session
   * @param {string} sessionId
   * @returns {Promise<number>} Number of deleted cache entries
   */
  async bulkInvalidateTranslationsBySession(sessionId) {
    return this._transaction(STORES.TRANSLATION_CACHE, 'readwrite', (tx) => {
      const store = tx.objectStore(STORES.TRANSLATION_CACHE);
      const index = store.index('sessionId');
      const request = index.openCursor(IDBKeyRange.only(sessionId));
      let count = 0;

      return new Promise((resolve, reject) => {
        request.onsuccess = (event) => {
          const cursor = event.target.result;
          if (cursor) {
            cursor.delete();
            count++;
            cursor.continue();
          } else {
            resolve(count);
          }
        };
        request.onerror = () => reject(request.error);
      });
    });
  }

  /**
   * Bulk invalidate all translation cache entries for a specific target language
   * @param {string} targetLang e.g. 'es', 'fr'
   * @returns {Promise<number>} Number of deleted entries
   */
  async bulkInvalidateTranslationsByLanguage(targetLang) {
    return this._transaction(STORES.TRANSLATION_CACHE, 'readwrite', (tx) => {
      const store = tx.objectStore(STORES.TRANSLATION_CACHE);
      const index = store.index('targetLang');
      const request = index.openCursor(IDBKeyRange.only(targetLang));
      let count = 0;

      return new Promise((resolve, reject) => {
        request.onsuccess = (event) => {
          const cursor = event.target.result;
          if (cursor) {
            cursor.delete();
            count++;
            cursor.continue();
          } else {
            resolve(count);
          }
        };
        request.onerror = () => reject(request.error);
      });
    });
  }

  /**
   * Delete all expired translation cache entries across all sessions
   * @returns {Promise<number>} Number of expired entries removed
   */
  async bulkDeleteExpiredCacheEntries() {
    const now = Date.now();
    return this._transaction(STORES.TRANSLATION_CACHE, 'readwrite', (tx) => {
      const store = tx.objectStore(STORES.TRANSLATION_CACHE);
      const index = store.index('expiresAt');
      const request = index.openCursor(IDBKeyRange.upperBound(now));
      let count = 0;

      return new Promise((resolve, reject) => {
        request.onsuccess = (event) => {
          const cursor = event.target.result;
          if (cursor) {
            cursor.delete();
            count++;
            cursor.continue();
          } else {
            resolve(count);
          }
        };
        request.onerror = () => reject(request.error);
      });
    });
  }

  // ────────────────────────────────────────────────
  // Example usage integration (add to your existing methods)
  // ────────────────────────────────────────────────

  // When deleting a session (example)
  async deleteSession(sessionId) {
    await this.bulkDeleteMessagesBySession(sessionId);
    await this.bulkInvalidateTranslationsBySession(sessionId);
    // ... delete metadata ...
    showToast(`Session and all related cache pruned — lattice lighter ⚡️`);
  }

  // On language change (already partially there)
  // ... after changing targetLang ...
  await this.bulkInvalidateTranslationsByLanguage(previousLang);
  await this.bulkDeleteExpiredCacheEntries(); // clean up

  // ... keep all previous methods (getCachedTranslation, cacheTranslation, etc.) ...
}

export const rathorDB = new RathorIndexedDB();    const allLatencies = Object.values(metrics.perLang).flatMap(d => d.latencies || []);
    const overallAvg = allLatencies.length > 0 ? Math.round(allLatencies.reduce((sum, l) => sum + l, 0) / allLatencies.length) : 0;
    const overallP95 = allLatencies.length > 0 ? Math.round(allLatencies.sort((a,b)=>a-b)[Math.floor(allLatencies.length * 0.95)]) : 0;

    return {
      total: metrics.total,
      hits: metrics.hits,
      misses: metrics.misses,
      hitRate: overallHitRate,
      avgLatency: overallAvg,
      p95Latency: overallP95,
      perLang: perLangStats,
      history: metrics.history || [],
      cacheSize: await this.getCacheSize()
    };
  }

  async updateTranslationMetrics(isHit, latencyMs = null, lang = targetTranslationLang) {
    const metrics = await this.getTranslationMetrics();
    metrics.total += 1;
    if (isHit) metrics.hits += 1;
    else metrics.misses += 1;

    metrics.perLang = metrics.perLang || {};
    metrics.perLang[lang] = metrics.perLang[lang] || { total: 0, hits: 0, misses: 0, latencies: [] };
    metrics.perLang[lang].total += 1;
    if (isHit) metrics.perLang[lang].hits += 1;
    else {
      metrics.perLang[lang].misses += 1;
      if (latencyMs !== null) {
        metrics.perLang[lang].latencies.push(latencyMs);
        if (metrics.perLang[lang].latencies.length > 500) metrics.perLang[lang].latencies.shift();
      }
    }

    metrics.history.push({
      timestamp: Date.now(),
      hit: isHit,
      hitRate: Math.round((metrics.hits / metrics.total) * 100),
      latency: latencyMs,
      lang
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

  // ... keep all previous translation cache methods ...
}

export const rathorDB = new RathorIndexedDB();      cacheSize: await this.getCacheSize()
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
