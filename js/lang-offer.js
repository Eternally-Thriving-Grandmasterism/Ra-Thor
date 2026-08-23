/* js/lang-offer.js — privacy-safe language offer
 * Workspace 14.15.6 · TOLC 8 · info@Rathor.ai
 * Reads navigator.language on this device only.
 * Never sent to a server. Never overrides a saved choice.
 */
(function () {
  'use strict';

  var SUPPORTED = ['en', 'ar', 'es', 'fr', 'nl', 'de', 'zh', 'ja', 'pt', 'ru', 'hi'];
  var OFFER = {
    en: { title: 'Use English?', body: 'Your browser is set to English. Stay in English, or pick another language above.', yes: 'Use English', no: 'Keep this page' },
    ar: { title: 'استخدام العربية؟', body: 'يبدو أن لغة المتصفح عربية. يمكننا عرض را-ثور بالعربية على هذا الجهاز فقط.', yes: 'استخدم العربية', no: 'إبقاء الصفحة كما هي' },
    es: { title: '¿Usar español?', body: 'Tu navegador está en español. Podemos mostrar Ra-Thor en español solo en este dispositivo.', yes: 'Usar español', no: 'Dejar esta página' },
    fr: { title: 'Utiliser le français ?', body: 'Votre navigateur est en français. Nous pouvons afficher Ra-Thor en français sur cet appareil uniquement.', yes: 'Utiliser le français', no: 'Garder cette page' },
    nl: { title: 'Nederlands gebruiken?', body: 'Je browser staat op Nederlands. We kunnen Ra-Thor alleen op dit apparaat in het Nederlands tonen.', yes: 'Nederlands gebruiken', no: 'Deze pagina houden' },
    de: { title: 'Deutsch verwenden?', body: 'Ihr Browser ist auf Deutsch. Ra-Thor kann nur auf diesem Gerät auf Deutsch angezeigt werden.', yes: 'Deutsch verwenden', no: 'Diese Seite behalten' },
    zh: { title: '使用简体中文？', body: '检测到浏览器语言为中文。仅在本设备上切换，不会发送到服务器。', yes: '使用中文', no: '保持当前页面' },
    ja: { title: '日本語にしますか？', body: 'ブラウザの言語が日本語です。この端末だけ日本語にできます。サーバーには送りません。', yes: '日本語にする', no: 'このページのまま' },
    pt: { title: 'Usar português?', body: 'O seu navegador está em português. Podemos mostrar o Ra-Thor em português só neste dispositivo.', yes: 'Usar português', no: 'Manter esta página' },
    ru: { title: 'Использовать русский?', body: 'Язык браузера — русский. Можно показать Ra-Thor по-русски только на этом устройстве.', yes: 'Использовать русский', no: 'Оставить эту страницу' },
    hi: { title: 'हिन्दी उपयोग करें?', body: 'ब्राउज़र की भाषा हिन्दी लगती है। केवल इस डिवाइस पर बदलेंगे — सर्वर पर नहीं भेजेंगे।', yes: 'हिन्दी उपयोग करें', no: 'यह पृष्ठ रखें' }
  };

  function stored() {
    try { return localStorage.getItem('rathor-lang') || ''; } catch (e) { return ''; }
  }
  function declined() {
    try { return localStorage.getItem('rathor-lang-offer') === 'declined'; } catch (e) { return false; }
  }
  function mapLang(tag) {
    if (!tag) return '';
    var short = String(tag).toLowerCase().replace('_', '-').split('-')[0];
    if (short === 'zh') return 'zh';
    if (SUPPORTED.indexOf(short) !== -1) return short;
    return '';
  }
  function guess() {
    var list = [];
    try {
      if (navigator.languages && navigator.languages.length) {
        for (var i = 0; i < navigator.languages.length; i++) list.push(navigator.languages[i]);
      }
    } catch (e) {}
    if (navigator.language) list.push(navigator.language);
    for (var j = 0; j < list.length; j++) {
      var hit = mapLang(list[j]);
      if (hit) return hit;
    }
    return 'en';
  }
  function applyLang(lang) {
    if (!lang || SUPPORTED.indexOf(lang) === -1) return;
    try { localStorage.setItem('rathor-lang', lang); } catch (e) {}
    if (typeof window.switchLanguage === 'function') { try { window.switchLanguage(lang); } catch (e) {} }
    if (typeof window.switchContactLang === 'function') { try { window.switchContactLang(lang); } catch (e) {} }
    if (typeof window.applyLockI18n === 'function') { try { window.applyLockI18n(lang); } catch (e) {} }
    document.documentElement.setAttribute('lang', lang);
    document.documentElement.setAttribute('dir', lang === 'ar' ? 'rtl' : 'ltr');
    document.querySelectorAll('.lang-tab, [data-lang]').forEach(function (btn) {
      var code = btn.getAttribute && btn.getAttribute('data-lang');
      if (!code) return;
      if (code === lang) btn.classList.add('active');
      else btn.classList.remove('active');
    });
  }
  function closeOffer() {
    var el = document.getElementById('rathor-lang-offer');
    if (el && el.parentNode) el.parentNode.removeChild(el);
  }
  function showOffer(lang) {
    if (!document.body || document.getElementById('rathor-lang-offer')) return;
    var t = OFFER[lang] || OFFER.en;
    var wrap = document.createElement('div');
    wrap.id = 'rathor-lang-offer';
    wrap.setAttribute('role', 'dialog');
    wrap.dir = lang === 'ar' ? 'rtl' : 'ltr';
    wrap.className = 'fixed bottom-4 left-4 right-4 sm:left-6 sm:right-auto sm:max-w-sm z-[9998] bg-zinc-950 border border-amber-300/35 rounded-2xl p-4 text-sm text-white/80 shadow-2xl';
    wrap.innerHTML =
      '<p class="text-amber-200 font-semibold mb-1">' + t.title + '</p>' +
      '<p class="text-white/65 text-[13px] leading-relaxed">' + t.body + '</p>' +
      '<p class="text-[11px] text-white/40 mt-2">On this device only. Not sent to rathor.ai.</p>' +
      '<div class="mt-3 flex flex-wrap gap-2">' +
      '<button type="button" id="rathor-lang-yes" class="px-4 py-2 rounded-xl bg-amber-300 text-black text-sm font-semibold">' + t.yes + '</button>' +
      '<button type="button" id="rathor-lang-no" class="px-4 py-2 rounded-xl border border-white/20 text-white/70 text-sm">' + t.no + '</button>' +
      '</div>';
    document.body.appendChild(wrap);
    var yes = document.getElementById('rathor-lang-yes');
    var no = document.getElementById('rathor-lang-no');
    if (yes) yes.addEventListener('click', function () { applyLang(lang); closeOffer(); });
    if (no) no.addEventListener('click', function () {
      try { localStorage.setItem('rathor-lang-offer', 'declined'); } catch (e) {}
      closeOffer();
    });
    setTimeout(closeOffer, 28000);
  }
  function boot() {
    var saved = stored();
    if (saved && SUPPORTED.indexOf(saved) !== -1) { applyLang(saved); return; }
    var g = guess();
    var page = (document.documentElement.getAttribute('lang') || 'en').slice(0, 2);
    if (g === page) return;
    if (declined()) return;
    if (SUPPORTED.indexOf(g) === -1) return;
    showOffer(g);
  }
  if (document.body) boot();
  else document.addEventListener('DOMContentLoaded', boot);
})();
