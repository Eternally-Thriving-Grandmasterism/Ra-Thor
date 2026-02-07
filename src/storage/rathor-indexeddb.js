/**
 * Rathor-NEXi IndexedDB Schema & Encrypted Storage Layer (v6 – current stable)
 * 
 * Encryption added: AES-GCM-256 via Web Crypto API
 * - Sensitive fields (content, translatedText, name, description, tags) encrypted
 * - Key derived from user passphrase (PBKDF2 + salt)
 * - Passphrase held in memory only — lost on page reload
 * - Non-sensitive fields (id, sessionId, timestamp, targetLang) in cleartext for indexing
 */

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

const CACHE_TTL_MS = 7 * 24 * 60 * 60 * 1000; // 7 days

// Encryption constants
const ENC_ALGORITHM = 'AES-GCM';
const ENC_KEY_LENGTH = 256;
const PBKDF2_ITERATIONS = 100000;
const PBKDF2_HASH = 'SHA-256';
const SALT_LENGTH = 16;
const IV_LENGTH = 12;

// In-memory encryption key (set once per session)
let encryptionKey = null;
let encryptionSalt = null;

// ────────────────────────────────────────────────
// Encryption / Decryption Helpers
// ────────────────────────────────────────────────

async function deriveKey(passphrase, salt) {
  const encoder = new TextEncoder();
  const keyMaterial = await crypto.subtle.importKey(
    'raw',
    encoder.encode(passphrase),
    'PBKDF2',
    false,
    ['deriveBits', 'deriveKey']
  );

  return crypto.subtle.deriveKey(
    {
      name: 'PBKDF2',
      salt,
      iterations: PBKDF2_ITERATIONS,
      hash: PBKDF2_HASH
    },
    keyMaterial,
    { name: ENC_ALGORITHM, length: ENC_KEY_LENGTH },
    false,
    ['encrypt', 'decrypt']
  );
}

async function encryptData(data) {
  if (!encryptionKey) throw new Error('Encryption key not set — passphrase required');

  const encoder = new TextEncoder();
  const encoded = encoder.encode(JSON.stringify(data));
  const iv = crypto.getRandomValues(new Uint8Array(IV_LENGTH));

  const encrypted = await crypto.subtle.encrypt(
    { name: ENC_ALGORITHM, iv },
    encryptionKey,
    encoded
  );

  return {
    iv: Array.from(iv),
    ciphertext: Array.from(new Uint8Array(encrypted))
  };
}

async function decryptData(encryptedObj) {
  if (!encryptionKey) throw new Error('Encryption key not set — passphrase required');

  const iv = new Uint8Array(encryptedObj.iv);
  const ciphertext = new Uint8Array(encryptedObj.ciphertext);

  const decrypted = await crypto.subtle.decrypt(
    { name: ENC_ALGORITHM, iv },
    encryptionKey,
    ciphertext
  );

  const decoder = new TextDecoder();
  return JSON.parse(decoder.decode(decrypted));
}

// ────────────────────────────────────────────────
// Passphrase / Key Initialization
// ────────────────────────────────────────────────

/**
 * Initialize encryption key from user passphrase
 * Called once on first launch or after reload
 */
async function initEncryption(passphrase) {
  if (!passphrase) throw new Error('Passphrase required for encryption');

  encryptionSalt = crypto.getRandomValues(new Uint8Array(SALT_LENGTH));
  encryptionKey = await deriveKey(passphrase, encryptionSalt);

  // Store salt (not secret) for future sessions
  localStorage.setItem('rathor_enc_salt', JSON.stringify(Array.from(encryptionSalt)));
}

/**
 * Load existing salt and prompt for passphrase if needed
 */
async function loadOrInitEncryption() {
  const savedSalt = localStorage.getItem('rathor_enc_salt');
  if (savedSalt) {
    encryptionSalt = new Uint8Array(JSON.parse(savedSalt));
  } else {
    // First time — generate salt & prompt passphrase
    encryptionSalt = crypto.getRandomValues(new Uint8Array(SALT_LENGTH));
    localStorage.setItem('rathor_enc_salt', JSON.stringify(Array.from(encryptionSalt)));
  }

  // Prompt user for passphrase (in real UI this would be a modal)
  const passphrase = prompt('Enter your Rathor encryption passphrase (saved in memory only):');
  if (!passphrase) throw new Error('Passphrase required');

  encryptionKey = await deriveKey(passphrase, encryptionSalt);
}

// Call on app init
window.addEventListener('load', async () => {
  try {
    await loadOrInitEncryption();
    console.log('[Rathor IndexedDB] Encryption initialized');
  } catch (err) {
    console.error('[Rathor IndexedDB] Encryption setup failed:', err);
    alert('Encryption passphrase required — application will run in limited mode.');
  }
});

// ────────────────────────────────────────────────
// Encrypted wrappers for sensitive operations
// ────────────────────────────────────────────────

async function encryptedPut(storeName, record) {
  const encryptedRecord = { ...record };

  if (storeName === STORES.CHAT_HISTORY) {
    encryptedRecord.content = await encryptData(record.content);
  } else if (storeName === STORES.TRANSLATION_CACHE) {
    encryptedRecord.translatedText = await encryptData(record.translatedText);
  } else if (storeName === STORES.SESSION_METADATA) {
    encryptedRecord.name = await encryptData(record.name);
    encryptedRecord.description = await encryptData(record.description || '');
    encryptedRecord.tags = await encryptData(record.tags || []);
  }

  return this._transaction(storeName, 'readwrite', ({ store }) => {
    store.put(encryptedRecord);
  });
}

async function encryptedGet(storeName, key) {
  const record = await this._transaction(storeName, 'readonly', ({ store }) => store.get(key));

  if (!record) return null;

  if (storeName === STORES.CHAT_HISTORY) {
    record.content = await decryptData(record.content);
  } else if (storeName === STORES.TRANSLATION_CACHE) {
    record.translatedText = await decryptData(record.translatedText);
  } else if (storeName === STORES.SESSION_METADATA) {
    record.name = await decryptData(record.name);
    record.description = await decryptData(record.description);
    record.tags = await decryptData(record.tags);
  }

  return record;
}

// ────────────────────────────────────────────────
// Bulk operations — now using encrypted wrappers
// ────────────────────────────────────────────────

async bulkSaveMessages(messages, onProgress = null) {
  if (!Array.isArray(messages) || messages.length === 0) return 0;

  let savedCount = 0;

  await this._transaction(STORES.CHAT_HISTORY, 'readwrite', async ({ store, tx }) => {
    for (const msg of messages) {
      if (!msg.id) msg.id = crypto.randomUUID();
      msg.sessionId = this.activeSessionId;

      const encryptedMsg = { ...msg };
      encryptedMsg.content = await encryptData(msg.content);

      store.put(encryptedMsg);
      savedCount++;

      if (onProgress && savedCount % 10 === 0) {
        onProgress(Math.round(savedCount / messages.length * 100), `Encrypted & saved \( {savedCount}/ \){messages.length} messages...`);
      }
    }
  }, onProgress);

  return savedCount;
}

// Similar encryption wrappers for bulk delete & invalidate (metadata only, no need to decrypt for delete)

// ... rest of the class (open, transaction helper, previous bulk methods) kept intact ...
