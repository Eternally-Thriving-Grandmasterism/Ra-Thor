/* family-nav-i18n.js — apply rathor.ai locale packs to the family bar.
   Workspace 14.15.6 · 2026-09-03 · info@Rathor.ai
*/
(function () {
  var HREF_KEYS = {
    '/': 'navHome',
    '/index.html': 'navHome',
    '/chat.html': 'navChat',
    '/Launch-Ra-Thor.html': 'navLaunch',
    '/micro-moment.html': 'navMoments',
    '/sovereign-shard.html': 'navShard',
    '/web-forge.html': 'navForge',
    '/contact.html': 'navContact',
    '/privacy.html': 'navPrivacy'
  };

  function packFor(lang) {
    var all = window.translations || {};
    return all[lang] || all.en || {};
  }

  function val(pack, key) {
    if (pack[key] !== undefined) return pack[key];
    var en = (window.translations && window.translations.en) || {};
    return en[key];
  }

  window.applyFamilyNavI18n = function (lang) {
    var pack = packFor(lang);
    var nav = document.getElementById('rt-family-nav');
    if (nav) {
      var links = nav.querySelectorAll('a[href]');
      for (var i = 0; i < links.length; i++) {
        var href = links[i].getAttribute('href') || '';
        var key = HREF_KEYS[href];
        var label = key ? val(pack, key) : null;
        if (label) links[i].textContent = label;
      }
    }
    var skip = document.getElementById('rt-skip-family');
    var skipLabel = val(pack, 'navSkip');
    if (skip && skipLabel) skip.textContent = skipLabel;

    document.querySelectorAll('[data-rt-family-footer] [data-i18n], .rt-site-footer [data-i18n]').forEach(function (el) {
      var key = el.getAttribute('data-i18n');
      var text = val(pack, key);
      if (text === undefined) return;
      if (el.hasAttribute('data-i18n-html')) el.innerHTML = text;
      else el.textContent = text;
    });

    var rtl = ['ar', 'fa', 'he', 'ur'].indexOf(lang) !== -1;
    try {
      document.documentElement.setAttribute('lang', lang || 'en');
      document.documentElement.setAttribute('dir', rtl ? 'rtl' : 'ltr');
    } catch (e) {}
  };

  document.addEventListener('rathor-nav-ready', function () {
    var lang = 'en';
    try { lang = localStorage.getItem('rathor-lang') || lang; } catch (e) {}
    if (typeof window.applyFamilyNavI18n === 'function') window.applyFamilyNavI18n(lang);
  });
})();
