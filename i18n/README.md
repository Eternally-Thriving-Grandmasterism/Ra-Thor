# Rathor.ai homepage locale packs

**Cache token:** `20260915b`  
**Source of truth:** `i18n/en.js` (living keys + Follow + claim lock + week lines)  
**Contact:** info@Rathor.ai

Buttons on `index.html` load `/i18n/{lang}.js?v=20260915b`. Missing keys fall back to English in the page script. Never blank. Never invent METR. RTL: `ar`, `fa`, `he`.

W8 completeness (every pack): family labels (`navHome`…`navPrivacy`) · Follow labels · hero/subhead (`headline`, `fusion`, `kicker`) · week card (`weekTitle`, `weekLineRa`, `weekLinePowrush`, `weekLineResearch`, `weekMore`) · Employ primary CTA (`employCta`) · Contact inquiry (`contactInquiry`).

Service worker precaches `/i18n/*.js`. Offline packs are the product UX. Google Translate is a click-only opt-in (`js/google-translate-optin.js`) and is **not** loaded on first paint.

| Code | Language | Notes |
| --- | --- | --- |
| en | English | Living voice 2026-09-15. Claim lock. Headline is the brand word. |
| ar | العربية | RTL |
| es | Español | |
| fr | Français | faqA8 is NOT “RBE already here” |
| nl | Nederlands | |
| de | Deutsch | |
| zh | 简体中文 | |
| ja | 日本語 | |
| pt | Português | |
| ru | Русский | |
| hi | हिन्दी | |
| it | Italiano | |
| ko | 한국어 | |
| uk | Українська | |
| pl | Polski | |
| tr | Türkçe | |
| vi | Tiếng Việt | |
| id | Bahasa Indonesia | |
| sv | Svenska | |
| th | ไทย | |
| el | Ελληνικά | |
| fa | فارسی | RTL |
| he | עברית | RTL |

Do not copy the pre-2026-08-31 “RBE royalties dissolve into abundance” FAQ line. `faqA8` on every pack must stay: RBE is design intent, not a present economic fact.

AG-SML v1.1. Independent of xAI. Not certified aircraft, plants, chains, or a replaced money system.
