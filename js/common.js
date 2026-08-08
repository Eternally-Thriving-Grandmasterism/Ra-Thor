// js/common.js — Shared utilities & i18n across all Rathor-NEXi pages
// Fully wired language tabs + expanded practical support + solid fallbacks

const registryUrl = '/locales/languages.json';

// Expanded practical supported list (matches the tabs on index.html + existing files)
const supported = [
  'en','ar','es','fr','de','nl','it','pt','ru','uk','pl','tr',
  'ja','ko','zh','hi','id','vi','th','sw','fa','he'
];

// ────────────────────────────────────────────────
// i18n Initialization & Language Management
// ────────────────────────────────────────────────

async function initI18n() {
  await i18next.init({
    lng: localStorage.getItem('rathor_lang') || getBestLanguage(),
    fallbackLng: 'en',
    debug: false,
    ns: 'translation',
    defaultNS: 'translation',
    interpolation: { escapeValue: false }
  });

  try {
    const registryResp = await fetch(registryUrl);
    const registry = await registryResp.json();
    registry.languages.forEach(lang => {
      i18next.addResourceBundle(lang.code, 'translation', {}, true, true);
    });
  } catch (e) {
    console.warn('Could not load languages registry', e);
  }

  await loadLanguage(i18next.language);
  updateContent();
  applyRTL(i18next.language);
  updateActiveLangButton(i18next.language);
}

// Load specific language JSON dynamically + cache it (with solid fallback)
async function loadLanguage(lng) {
  if (i18next.services.resourceStore.hasResourceBundle(lng, 'translation') && 
      Object.keys(i18next.services.resourceStore.data[lng]?.translation || {}).length > 0) {
    return;
  }

  try {
    const cache = await caches.open('rathor-locales');
    const cached = await cache.match(`/locales/${lng}.json`);

    if (cached) {
      const json = await cached.json();
      i18next.addResourceBundle(lng, 'translation', json, true, true);
      return;
    }

    const response = await fetch(`/locales/${lng}.json`);
    if (!response.ok) throw new Error(`Locale not found: ${lng}`);
    const json = await response.json();

    await cache.put(`/locales/${lng}.json`, new Response(JSON.stringify(json), {
      headers: { 'Content-Type': 'application/json' }
    }));

    i18next.addResourceBundle(lng, 'translation', json, true, true);
  } catch (err) {
    console.warn(`Failed to load ${lng}, falling back to English`, err);
    if (lng !== 'en') {
      await loadLanguage('en');
    }
  }
}

// Change language + update UI + RTL + active button + toast
async function changeLanguage(lng) {
  await loadLanguage(lng);
  await i18next.changeLanguage(lng);
  updateContent();
  applyRTL(lng);
  updateActiveLangButton(lng);
  localStorage.setItem('rathor_lang', lng);
  showToast(`Language switched to ${lng.toUpperCase()} ⚡️`);
}

// Apply RTL layout for right-to-left languages
function applyRTL(lng) {
  const rtlLanguages = ['ar', 'he', 'fa', 'ur'];
  const isRTL = rtlLanguages.includes(lng);
  document.body.classList.toggle('rtl', isRTL);
  document.documentElement.setAttribute('dir', isRTL ? 'rtl' : 'ltr');
}

// Highlight the correct language button
function updateActiveLangButton(lng) {
  document.querySelectorAll('.lang-tab').forEach(btn => {
    btn.classList.toggle('active', btn.getAttribute('data-lang') === lng);
  });
}

// Update all translatable elements on page
function updateContent() {
  document.querySelectorAll('[data-i18n]').forEach(el => {
    const key = el.getAttribute('data-i18n');
    if (key) el.innerHTML = i18next.t(key);
  });
}

// Best initial language from browser preferences
function getBestLanguage() {
  const preferred = navigator.languages || [navigator.language];
  for (const lang of preferred) {
    const short = lang.split('-')[0].toLowerCase();
    if (supported.includes(short)) return short;
  }
  return 'en';
}

// Shared toast utility
function showToast(message) {
  const existing = document.querySelector('.rathor-toast');
  if (existing) existing.remove();

  const toast = document.createElement('div');
  toast.className = 'rathor-toast';
  toast.textContent = message;
  toast.style.cssText = `
    position: fixed; bottom: 80px; left: 50%; transform: translateX(-50%);
    background: #fcd34d; color: #000; padding: 12px 24px;
    border-radius: 12px; box-shadow: 0 4px 20px rgba(252, 211, 77, 0.5);
    z-index: 4000; font-weight: 600; white-space: pre-wrap; max-width: 90%;
  `;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 3500);
}

// ────────────────────────────────────────────────
// Wire language buttons (this was missing)
// ────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  // Attach click handlers to all language tabs
  document.querySelectorAll('.lang-tab[data-lang]').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.preventDefault();
      const lng = btn.getAttribute('data-lang');
      if (lng) changeLanguage(lng);
    });
  });

  // Initialize i18n
  if (typeof i18next !== 'undefined') {
    initI18n();
  } else {
    // Fallback if i18next is loaded later
    window.addEventListener('load', () => {
      if (typeof i18next !== 'undefined') initI18n();
    });
  }
});

// Export shared utilities
window.rathorCommon = {
  initI18n,
  loadLanguage,
  changeLanguage,
  applyRTL,
  updateContent,
  getBestLanguage,
  showToast,
  updateActiveLangButton
};
