# Rathor.ai homepage locale packs

**Cache token:** `20260924a` (`i18n-chrome.js`, `i18n-essay.js`, and packs)  
**Source of truth:** `i18n/en.js`  
**Contact:** info@Rathor.ai

## Visitor essays are packed (W4)

Offline packs apply **chrome** and **visitor essays**. Operator documents (`docs/**`) stay English.

**Chrome** (`js/i18n-chrome.js`):

- family labels — `navHome` … `navPrivacy`
- Follow — `followTitle`, `followX`, `followLinkedIn`, `followFacebook`
- hero / subhead — `headline`, `fusion`, `kicker`
- recent updates — `weekTitle`, `weekLineWrap`, `weekLineRa`, `weekLinePowrush`, `weekLineResearch`, `weekMore`
- session cards — titles, subtitles, CTAs (`grok*` `x*` `vibe*` `employCta`)
- Employ chrome — `employTitle`, `employSubtitle`
- install chip — `installTitle`, `installStatus`, `installCta`, `demoNote`
- steward line — `steward`
- language-storage note — `langStoredNote`
- homepage product-path cards — `pathPlayTitle`, `pathPlayBody`, `pathLicenseTitle`, `pathLicenseBody`, `pathInspectTitle`, `pathInspectBody`, `pathOfflineTitle`, `pathOfflineBody`, `pathWrapTitle`, `pathWrapBody`
- on-device cards — `surfacesTitle`, `surfaceChat`, `surfaceChatNote`, `surfaceMap`, `surfaceMapNote`, `surfaceShard`, `surfaceShardNote`, `surfaceRepo`, `surfaceRepoNote`, `homeSurfacePaper`, `homeSurfacePaperNote`
- Contact inquiry — `contactInquiry`
- Google tab chrome — `gTranslateBtn`, `gTranslateNote`
- Lattice Chat labels — `chatTitle`, `chatSubtitle`, `chatOfflineMercy`, `chatPathFast`, `chatPathServer`, `chatLocalIntel`, `chatStatusDefault`, `chatNotAvailable`, `chatLocalNote`, `chatSearch`, `chatSpeak`, `chatSend`, `chatSessionFoot`, `chatBridgeTitle`, `chatBridgeBody`, `chatCopyTitle`, `chatCopyContext`, `chatOpenGrok`, `chatOpenX`
- Lattice Chat canned Fast Responder lines — `chatReplyHello` through `chatReplyFallback`

Lattice Chat chrome and canned lines are translated in the other 22 packs. The Fast Responder is not a translator and does not speak 23 languages. When `rathor-lang` is set, Copy Context and the Local Server / WebLLM system preamble may add `Reply in {language name}.` That line is for the optional model. Canned replies stay the pack string.

`#chat-messages` and `#chat-input` follow the applied string’s script. Arabic, Persian, or Hebrew script sets `dir=rtl` on those two nodes. An English copy stays `dir=ltr`. Family pills and language tabs stay left-to-right.

**Visitor essays** (`js/i18n-essay.js`, loaded after chrome):

- FAQ questions and answers (`faqQ*`, `faqA*`, `faqTitle`, `faqContact`)
- Employ article A–G (`article.rt-prose`)
- Privacy sections
- Briefing body
- Contact guidance and the commercial-inquiry form labels
- Launch, go-x, and science-watch short prose

FAQ keys that already had a translation still apply. The 272 visitor essay keys stay packed. This chat seat does not retranslate them. A missing key falls back to English, so the node stays filled. Filenames, emails, version tokens, and regulation acronyms stay Latin when that is the whole value.

Missing key → English. Never blank. Never invent METR. Recent Updates lines stay chrome, including `weekLineResearch`, and stay undated. `i18n-chrome.js` stamps family-nav and Follow labels, and mounts `#lang-selector` when a page has none. Site-lock delegates to chrome apply; essay apply follows that event. The family pill row and the language tabs stay left-to-right.

## Direction

`dir=rtl` only when the **applied string** for that node is actually RTL (`ar` / `fa` / `he` script). English fallback forces `dir=ltr lang=en` on that node. If most chrome nodes fell back, `<main>` stays `ltr`. `html[dir]` follows chrome only — not the language code alone, and not the essay. When a pack’s essay string differs from English, that `<article>` / `#faq` / `[data-rt-prose]` takes the applied language (`dir=rtl` only if the applied essay text is RTL script). An English copy keeps `dir=ltr lang=en`. Family pill row stays Home…Privacy left-to-right (do not mirror). Language tabs stay left-to-right.

Buttons on `index.html` load `/i18n/{lang}.js?v=20260924a`. `chat.html` loads the same token.

Service worker precaches every `/i18n/*.js` pack plus `/js/i18n-chrome.js` and `/js/i18n-essay.js`. Script fetches stay network-first. If the network fails, `/i18n/*.js`, `i18n-chrome.js`, and `i18n-essay.js` match the precache with the query string ignored, so `?v=20260924a` still opens offline.

Google Translate is a new tab (`js/google-translate-optin.js`). Chat keeps COEP. Marketing pages are fetchable by Google. Offline packs remain the default.

| Code | Language | Notes |
| --- | --- | --- |
| en | English | Living voice 2026-09-15. Claim lock. Headline is the brand word. |
| ar | العربية | RTL only when the applied string is actually Arabic |
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
| fa | فارسی | RTL only when the applied string is actually Persian |
| he | עברית | RTL only when the applied string is actually Hebrew |

Do not copy the pre-2026-08-31 “RBE royalties dissolve into abundance” FAQ line. `faqA8` on every pack must stay: RBE is design intent, not a present economic fact. Essay apply fills a marked `faqA8` node from that pack line. The current homepage FAQ does not render question 8. The pack line must not state a resource-based economy as a present fact.

AG-SML v1.1. Independent of xAI. Not certified aircraft, plants, chains, or a replaced money system.
