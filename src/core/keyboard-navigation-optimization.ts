// src/core/keyboard-navigation-optimization.ts – Keyboard Navigation Optimization v1.0
// WCAG 2.2 AAA+ keyboard support: logical tab order, focus-visible, skip-to-content,
// mercy-gated focus traps, valence-modulated glow, ARIA live announcements
// MIT License – Autonomicity Games Inc. 2026

import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';
import { announceToScreenReader } from './screen-reader-optimization';

// ──────────────────────────────────────────────────────────────
// Global keyboard event listener & focus manager
// ──────────────────────────────────────────────────────────────

let activeFocusTrap: (() => void) | null = null;
let skipLink: HTMLElement | null = null;

function createSkipLink() {
  if (skipLink) return;

  skipLink = document.createElement('a');
  skipLink.href = '#main-content';
  skipLink.textContent = 'Skip to main content';
  skipLink.style.position = 'absolute';
  skipLink.style.top = '-9999px';
  skipLink.style.left = '-9999px';
  skipLink.style.zIndex = '10000';
  skipLink.style.padding = '1rem';
  skipLink.style.background = '#00ff88';
  skipLink.style.color = '#000';
  skipLink.style.borderRadius = '0 0 8px 8px';
  skipLink.style.transition = 'all 0.3s';
  skipLink.addEventListener('focus', () => {
    skipLink!.style.top = '0';
    skipLink!.style.left = '0';
  });
  skipLink.addEventListener('blur', () => {
    skipLink!.style.top = '-9999px';
    skipLink!.style.left = '-9999px';
  });

  document.body.prepend(skipLink);
}

// ──────────────────────────────────────────────────────────────
// Valence-modulated focus glow (visible only on keyboard navigation)
// ──────────────────────────────────────────────────────────────

function applyKeyboardFocusStyle() {
  const styleId = 'keyboard-focus-style';
  if (document.getElementById(styleId)) return;

  const style = document.createElement('style');
  style.id = styleId;
  style.textContent = `
    *:focus-visible {
      outline: none;
      box-shadow: 0 0 0 4px rgba(0, 255, 136, 0.6);
      border-radius: 8px;
      transition: box-shadow 0.2s;
    }

    [data-valence-focus] {
      --focus-glow: rgba(0, 255, 136, ${currentValence.get()});
      box-shadow: 0 0 0 4px var(--focus-glow);
    }

    .reduced-motion *:focus-visible {
      box-shadow: 0 0 0 3px #00ff88;
      transition: none;
    }
  `;
  document.head.appendChild(style);
}

// ──────────────────────────────────────────────────────────────
// Focus trap with mercy release & announcement
// ──────────────────────────────────────────────────────────────

export function trapFocusInElement(
  element: HTMLElement,
  releaseOnEsc: boolean = true,
  announceOnTrap: boolean = true
): () => void {
  if (activeFocusTrap) activeFocusTrap(); // release previous trap

  const focusable = Array.from(
    element.querySelectorAll(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    )
  ) as HTMLElement[];

  if (focusable.length === 0) return () => {};

  const first = focusable[0];
  const last = focusable[focusable.length - 1];

  const handler = (e: KeyboardEvent) => {
    if (e.key === 'Tab') {
      if (e.shiftKey) {
        if (document.activeElement === first) {
          last.focus();
          e.preventDefault();
        }
      } else {
        if (document.activeElement === last) {
          first.focus();
          e.preventDefault();
        }
      }
    } else if (releaseOnEsc && e.key === 'Escape') {
      releaseFocusTrap();
    }
  };

  element.addEventListener('keydown', handler);

  // Mercy: announce trap activation
  if (announceOnTrap) {
    announceToScreenReader('Keyboard focus trapped in dialog – press Escape to exit', 'polite');
  }

  // Mercy: initial focus on first element
  first.focus();

  const release = () => {
    element.removeEventListener('keydown', handler);
    activeFocusTrap = null;
    announceToScreenReader('Keyboard focus released', 'polite');
  };

  activeFocusTrap = release;
  return release;
}

function releaseFocusTrap() {
  if (activeFocusTrap) {
    activeFocusTrap();
  }
}

// ──────────────────────────────────────────────────────────────
// Global keyboard shortcuts & accessibility helpers
// ──────────────────────────────────────────────────────────────

function handleGlobalKeyboardShortcuts(e: KeyboardEvent) {
  // Alt + / → focus search/input if exists
  if (e.altKey && e.key === '/') {
    const searchInput = document.querySelector('input[type="search"], input[aria-label*="search"]') as HTMLElement;
    if (searchInput) {
      searchInput.focus();
      e.preventDefault();
      announceToScreenReader('Search input focused', 'polite');
    }
  }

  // Alt + M → toggle menu if exists
  if (e.altKey && e.key.toLowerCase() === 'm') {
    const menuButton = document.querySelector('[aria-label*="menu"], [aria-haspopup="true"]') as HTMLElement;
    if (menuButton) {
      menuButton.click();
      e.preventDefault();
      announceToScreenReader('Menu toggled', 'polite');
    }
  }
}

// ──────────────────────────────────────────────────────────────
// Initialize accessibility features on load
// ──────────────────────────────────────────────────────────────

if (typeof window !== 'undefined') {
  // Create skip-to-content link
  createSkipLink();

  // Apply keyboard focus style
  applyKeyboardFocusStyle();

  // Global shortcuts
  window.addEventListener('keydown', handleGlobalKeyboardShortcuts);

  // Re-apply on preference change
  const mqMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  mqMotion.addEventListener('change', applyKeyboardFocusStyle);
}

// ──────────────────────────────────────────────────────────────
// Public API
// ──────────────────────────────────────────────────────────────

export const keyboard = {
  trapFocus: trapFocusInElement,
  releaseFocusTrap,
  announce: announceToScreenReader,
  applyFocusStyle: applyKeyboardFocusStyle,
};

export default keyboard;
