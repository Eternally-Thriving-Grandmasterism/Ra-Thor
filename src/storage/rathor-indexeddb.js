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
        const db = event.target.result;
        const tx = event.target.transaction;
        const oldVersion = event.oldVersion || 0;

        const createOrUpdateStore = (name, options = {}, indexes = []) => {
          let store;
          if (db.objectStoreNames.contains(name)) {
            store = tx.objectStore(name);
          } else {
            store = db.createObjectStore(name, options);
          }
          indexes.forEach(([keyPath, unique = false]) => {
            if (!store.indexNames.contains(keyPath)) {
              store.createIndex(keyPath, keyPath, { unique });
            }
          });
          return store;
        };

        // Previous migrations (v1–v5) – kept as is
        if (oldVersion < 1) { /* ... */ }
        if (oldVersion < 2) { /* ... */ }
        if (oldVersion < 3) { /* ... */ }
        if (oldVersion < 4) { /* session-metadata */ }
        if (oldVersion < 5) { /* tags */ }
        if (oldVersion < 6) { /* translation_cache */ }

        // No new schema for metrics — stored in user_preferences
      };

      request.onblocked = () => {
        console.warn('[Rathor IndexedDB] Upgrade blocked — close other tabs');
      };
    });
  }

  // ────────────────────────────────────────────────
  // Translation Metrics Tracking
  // ────────────────────────────────────────────────

  async getTranslationMetrics() {
    const metrics = await this._transaction(STORES.USER_PREFERENCES, 'readonly', (tx) => {
      const store = tx.objectStore(STORES.USER_PREFERENCES);
      const req = store.get('translation_metrics');
      return new Promise((res, rej) => {
        req.onsuccess = () => res(req.result || { total: 0, hits: 0, misses: 0, history: [] });
        req.onerror = () => rej(req.error);
      });
    });

    const hitRate = metrics.total > 0 ? Math.round((metrics.hits / metrics.total) * 100) : 0;
    return { ...metrics, hitRate };
  }

  async updateTranslationMetrics(isHit) {
    const metrics = await this.getTranslationMetrics();
    metrics.total += 1;
    if (isHit) metrics.hits += 1;
    else metrics.misses += 1;

    // Keep last 1000 entries for history
    metrics.history.push({
      timestamp: Date.now(),
      hit: isHit,
      hitRate: Math.round((metrics.hits / metrics.total) * 100)
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

  // ────────────────────────────────────────────────
  // Enhanced translateText with cache + metrics
  // ────────────────────────────────────────────────

  async translateText(text, messageId, sessionId, fromLang = 'auto', toLang = targetTranslationLang) {
    if (!isTranslationEnabled) return text;

    const cached = await this.getCachedTranslation(sessionId, messageId, toLang);
    if (cached) {
      await this.updateTranslationMetrics(true);
      showToast('Translation retrieved from eternal lattice cache ⚡️');
      return cached.translatedText;
    }

    try {
      if (!translator) {
        showTranslationProgress('Downloading offline translation model (one-time, \~40MB)...');
        const { pipeline } = Xenova;
        translator = await pipeline('translation', 'Xenova/m2m100_418M-distilled', {
          progress_callback: (progress) => {
            if (progress.status === 'progress') {
              const percent = Math.round(progress.loaded / progress.total * 100);
              updateTranslationProgress(percent);
            }
          }
        });
        updateTranslationProgress(100, 'Offline translation lattice awakened ⚡️');
        setTimeout(hideTranslationProgress, 800);
      }

      const output = await translator(text, {
        src_lang: fromLang === 'auto' ? undefined : fromLang,
        tgt_lang: `to_${toLang}`
      });

      const translated = output[0].translation_text;
      await this.cacheTranslation(sessionId, messageId, toLang, translated);
      await this.updateTranslationMetrics(false); // miss → new translation

      return translated;
    } catch (err) {
      console.error('Translation error:', err);
      await this.updateTranslationMetrics(false);
      return text + ' [translation offline error]';
    }
  }

  // ... keep all previous methods (getCachedTranslation, cacheTranslation, invalidate*, clearExpiredCache, etc.) ...
}

export const rathorDB = new RathorIndexedDB();
