/**
 * Ra-Thor Design Tokens - Tailwind CSS Preset
 * Production-grade integration for web-forge
 */

const tokens = require('../tokens.json').RaThor;

module.exports = {
  theme: {
    extend: {
      colors: {
        brand: {
          primary: tokens.color.semantic.brand.primary.value,
          secondary: tokens.color.semantic.brand.secondary.value,
        },
        background: {
          primary: tokens.color.semantic.background.primary.value,
          secondary: tokens.color.semantic.background.secondary.value,
          elevated: tokens.color.semantic.background.elevated.value,
        },
        text: {
          primary: tokens.color.semantic.text.primary.value,
          secondary: tokens.color.semantic.text.secondary.value,
          muted: tokens.color.semantic.text.muted.value,
        },
        border: {
          default: tokens.color.semantic.border.default.value,
          strong: tokens.color.semantic.border.strong.value,
        },
      },
      fontFamily: {
        sans: tokens.typography.fontFamily.sans.value,
        display: tokens.typography.fontFamily.display.value,
      },
      fontSize: tokens.typography.fontSize,
      fontWeight: tokens.typography.fontWeight,
      spacing: tokens.spacing,
      borderRadius: tokens.borderRadius,
    },
  },
};