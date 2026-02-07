// js/common.js — Shared across all pages

async function loadLanguageRegistry() {
  const res = await fetch('/locales/languages.json');
  return await res.json();
}

async function initI18n() {
  await i18next.init({
    lng: localStorage.getItem('rathor_lang') || getBestLanguage(),
    fallbackLng: 'en',
    debug: false
  });

  const registry = await loadLanguageRegistry();
  registry.languages.forEach(lang => {
    i18next.addResourceBundle(lang.code, 'translation', {}, true, true);
  });

  await loadLanguage(i18next.language);
  updateContent();
}

// ... include changeLanguage, applyRTL, updateContent, getBestLanguage, initLanguageSearch, etc. ...
