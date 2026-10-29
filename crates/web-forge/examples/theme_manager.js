/**
 * Ra-Thor Theme Manager
 * Production-grade theme switching with persistence
 */

const THEME_KEY = 'rathor-theme';

const themes = {
  default: 'default',
  light: 'light'
};

export function setTheme(theme) {
  if (!themes[theme]) {
    console.warn(`Theme "${theme}" not found. Falling back to default.`);
    theme = 'default';
  }

  document.documentElement.setAttribute('data-theme', theme);
  localStorage.setItem(THEME_KEY, theme);
}

export function getCurrentTheme() {
  return localStorage.getItem(THEME_KEY) || 'default';
}

export function initTheme() {
  const savedTheme = getCurrentTheme();
  setTheme(savedTheme);
}

// Auto-initialize if in browser
if (typeof window !== 'undefined') {
  window.addEventListener('load', initTheme);
}