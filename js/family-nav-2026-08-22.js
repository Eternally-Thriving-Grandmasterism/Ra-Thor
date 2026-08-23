/* family-nav-2026-08-22.js
   Shared public-surface network for rathor.ai
   Workspace 14.15.6 · Lattice Chat v14.18.x
   Contact: info@Rathor.ai
   2026-08-23: Install chip + coherent family labels
   2026-08-23b: Professional site footer contract · no pill-bar footer
*/
(function () {
  if (window.__rtFamilyNav) return;
  window.__rtFamilyNav = true;

  var LINKS = [
    { href: '/', label: 'Home' },
    { href: '/chat.html', label: 'Chat' },
    { href: '/Launch-Ra-Thor.html', label: 'Launch' },
    { href: '/micro-moment.html', label: 'Moments' },
    { href: '/sovereign-shard.html', label: 'Shard' },
    { href: '/web-forge.html', label: 'Forge' },
    { href: '/contact.html', label: 'Contact' },
    { href: '/privacy.html', label: 'Privacy' }
  ];

  function norm(p) {
    if (!p || p === '') return '/';
    p = p.split('?')[0].split('#')[0];
    if (p.charAt(p.length - 1) === '/' && p.length > 1) p = p.slice(0, -1);
    if (p === '/index.html') return '/';
    return p;
  }

  var here = norm(location.pathname || '/');

  function isHere(href) {
    return norm(href) === here;
  }

  function reduceMotion() {
    try {
      return window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    } catch (e) {
      return false;
    }
  }

  function isStandalone() {
    return (
      (window.matchMedia && window.matchMedia('(display-mode: standalone)').matches) ||
      window.navigator.standalone === true
    );
  }

  function alreadyHasFamilyHeader() {
    if (document.querySelector('[data-rt-family-header]')) return true;
    var scope = document.querySelector('header') || document.body;
    if (!scope) return false;
    var found = {};
    var anchors = scope.querySelectorAll('a');
    for (var i = 0; i < anchors.length; i++) {
      var t = (anchors[i].textContent || '').replace(/\s+/g, ' ').trim();
      if (t) found[t] = true;
    }
    return !!(found.Home && found.Chat && (found.Launch || found.Shard || found.Forge));
  }

  function compactOffTop() {
    var flag = document.body && document.body.getAttribute('data-rt-family');
    if (flag === 'off-top' || flag === 'off') return true;
    return alreadyHasFamilyHeader();
  }

  function styleLink(el, current) {
    el.style.cssText = 'font:600 11px/1.2 system-ui,sans-serif;padding:0.35rem 0.65rem;border-radius:999px;text-decoration:none;border:1px solid var(--rt-line, rgba(240,211,106,0.34));color:var(--rt-gold,#f0d36a);background:transparent;cursor:pointer;';
    el.addEventListener('focus', function () {
      el.style.outline = '2px solid var(--rt-gold-hot,#f5c84a)';
      el.style.outlineOffset = '2px';
    });
    el.addEventListener('blur', function () {
      el.style.outline = 'none';
    });
    if (current) {
      el.setAttribute('aria-current', 'page');
      el.style.background = 'var(--rt-gold-hot,#f5c84a)';
      el.style.color = 'var(--rt-on-gold,#111)';
      el.style.borderColor = 'var(--rt-gold-hot,#f5c84a)';
    }
  }

  function bar(kind) {
    var nav = document.createElement('nav');
    nav.id = kind === 'top' ? 'rt-family-nav' : 'rt-family-footer';
    nav.setAttribute('aria-label', kind === 'top' ? 'Ra-Thor family' : 'Ra-Thor family footer');
    var blur = reduceMotion() ? 'none' : 'blur(8px)';
    nav.style.cssText = kind === 'top'
      ? 'position:sticky;top:0;z-index:40;display:flex;flex-wrap:wrap;gap:0.4rem;justify-content:center;align-items:center;padding:0.45rem 0.7rem;background:var(--rt-nav-bg,rgba(5,5,5,0.9));border-bottom:1px solid var(--rt-line,rgba(240,211,106,0.34));backdrop-filter:' + blur + ';'
      : 'display:none;';
    if (kind !== 'top') nav.className = 'rt-family-pills';

    LINKS.forEach(function (item) {
      var a = document.createElement('a');
      a.href = item.href;
      a.textContent = item.label;
      styleLink(a, isHere(item.href));
      nav.appendChild(a);
    });

    if (!isStandalone()) {
      var install = document.createElement('button');
      install.type = 'button';
      install.textContent = 'Install';
      install.setAttribute('aria-label', 'Install Ra-Thor on this device');
      styleLink(install, false);
      install.addEventListener('click', function () {
        if (typeof window.rathorTriggerPWAInstall === 'function') {
          window.rathorTriggerPWAInstall();
        }
      });
      nav.appendChild(install);
    }

    var theme = document.createElement('button');
    theme.type = 'button';
    theme.id = kind === 'top' ? 'rt-theme-toggle' : 'rt-theme-toggle-foot';
    theme.setAttribute('data-rt-theme-toggle', '1');
    theme.textContent = 'Light';
    theme.setAttribute('aria-label', 'Toggle light and dark theme');
    styleLink(theme, false);
    theme.addEventListener('click', function () {
      if (typeof window.rathorToggleTheme === 'function') window.rathorToggleTheme();
    });
    nav.appendChild(theme);
    return nav;
  }

  function siteFooter() {
    var wrap = document.createElement('footer');
    wrap.className = 'rt-site-footer';
    wrap.id = 'rt-family-footer';
    wrap.setAttribute('data-rt-family-footer', '1');
    wrap.setAttribute('aria-label', 'Ra-Thor site footer');
    wrap.innerHTML =
      '<div class="max-w-5xl mx-auto px-4 sm:px-6">' +
        '<div class="grid grid-cols-1 md:grid-cols-12 gap-8">' +
          '<div class="md:col-span-3">' +
            '<h4>Trademarks</h4>' +
            '<p class="rt-legal">Ra-Thor™ is a trademark of Autonomicity Games Inc.<br>Grok is a trademark of xAI. X is a trademark of X Corp.<br>Ra-Thor is independent — not affiliated with, sponsored by, or endorsed by xAI.</p>' +
          '</div>' +
          '<div class="md:col-span-3">' +
            '<h4>Privacy</h4>' +
            '<p class="rt-legal">This website collects no personal data. Computations stay in your browser. No cookies, tracking, or analytics we control.</p>' +
          '</div>' +
          '<div class="md:col-span-3">' +
            '<h4>Workspace</h4>' +
            '<p class="rt-legal">v14.15.6 source of truth · Lattice Chat v14.18 · Shard v8 local demo · AG-SML v1.0 · TOLC 8 · Capable · Bounded · Corrigible</p>' +
          '</div>' +
          '<div class="md:col-span-3">' +
            '<h4>Family</h4>' +
            '<div class="flex flex-col gap-2 text-xs">' +
              '<a href="/">Home</a>' +
              '<a href="/chat.html">Lattice Chat</a>' +
              '<a href="/Launch-Ra-Thor.html">Launch map</a>' +
              '<a href="/micro-moment.html">Micro-moments</a>' +
              '<a href="/sovereign-shard.html">Sovereign Shard</a>' +
              '<a href="/web-forge.html">Web-Forge</a>' +
              '<a href="/contact.html">Contact</a>' +
              '<a href="/privacy.html">Privacy</a>' +
              '<a href="https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor" target="_blank" rel="noopener">Monorepo</a>' +
              '<a href="mailto:info@Rathor.ai">info@Rathor.ai</a>' +
            '</div>' +
          '</div>' +
        '</div>' +
        '<div class="pt-8 mt-8 border-t border-amber-300/20 text-xs flex flex-col md:flex-row justify-between items-center gap-4">' +
          '<div>© 2026 Sherif Samy Botros — sole steward of Autonomicity Games Inc. &amp; AlphaProMega Air Foundation. TOLC 8 · APTD.</div>' +
          '<a href="mailto:info@Rathor.ai">info@Rathor.ai</a>' +
        '</div>' +
      '</div>';
    return wrap;
  }

  function skipLink() {
    if (document.getElementById('rt-skip-family')) return;
    var a = document.createElement('a');
    a.id = 'rt-skip-family';
    a.href = '#rt-family-main';
    a.textContent = 'Skip family navigation';
    a.style.cssText = 'position:absolute;left:-999px;top:0;background:#fbbf24;color:#000;padding:0.4rem 0.7rem;z-index:50;font:600 12px system-ui,sans-serif;';
    a.addEventListener('focus', function () {
      a.style.left = '0.5rem';
      a.style.top = '0.5rem';
    });
    a.addEventListener('blur', function () {
      a.style.left = '-999px';
    });
    document.body.insertBefore(a, document.body.firstChild);
    if (!document.getElementById('rt-family-main')) {
      var main = document.querySelector('main') || document.body.children[1] || document.body;
      if (main && main !== a) main.id = main.id || 'rt-family-main';
    }
  }

  function mount() {
    if (document.body && document.body.getAttribute('data-rt-family') === 'off') return;
    skipLink();
    if (!document.getElementById('rt-family-nav') && !compactOffTop()) {
      document.body.insertBefore(bar('top'), document.body.firstChild);
    }
    if (!document.querySelector('.rt-site-footer') && !document.querySelector('[data-rt-family-footer]')) {
      document.body.appendChild(siteFooter());
    }
    try { window.dispatchEvent(new Event('rathor-nav-ready')); } catch (e) {}
  }

  function lockForgeCopy() {
    if (here !== '/web-forge.html') return;
    try {
      document.title = 'Ra-Thor Web-Forge • workspace 14.15.6';
      var nodes = document.querySelectorAll('h1, p, div, title');
      for (var i = 0; i < nodes.length; i++) {
        var el = nodes[i];
        if (!el.childNodes || el.childNodes.length !== 1 || el.childNodes[0].nodeType !== 3) continue;
        var s = el.textContent || '';
        if (s.indexOf('v14.0.0') !== -1 || s.indexOf('ONE Organism') !== -1 || s.indexOf('64+') !== -1) {
          el.textContent = s
            .replace(/v14\.0\.0/g, 'workspace 14.15.6')
            .replace(/ONE Organism • Live/g, 'Local demo · not a product')
            .replace(/64\+ PATSAGi Councils/g, 'workspace 14.15.6');
        }
      }
    } catch (e) {}
  }

  function bootScienceMap() {
    if (document.querySelector('script[src*="science-map-lock"]')) return;
    var s = document.createElement('script');
    s.src = '/js/science-map-lock.js';
    s.defer = true;
    (document.head || document.documentElement).appendChild(s);
  }

  function bootTheme() {
    if (!document.querySelector('link[href*="rathor-theme.css"]')) {
      var l = document.createElement('link');
      l.rel = 'stylesheet';
      l.href = '/css/rathor-theme.css';
      (document.head || document.documentElement).appendChild(l);
    }
    if (typeof window.rathorToggleTheme !== 'function' && !document.querySelector('script[src*="rathor-theme.js"]')) {
      var s = document.createElement('script');
      s.src = '/js/rathor-theme.js';
      s.onload = function () {
        try { window.dispatchEvent(new Event('rathor-nav-ready')); } catch (e) {}
      };
      (document.head || document.documentElement).appendChild(s);
    }
    if (!document.querySelector('script[src*="rathor-unify.js"]')) {
      var u = document.createElement('script');
      u.src = '/js/rathor-unify.js';
      (document.head || document.documentElement).appendChild(u);
    }
  }

  if (document.body) {
    bootTheme();
    mount();
    lockForgeCopy();
    bootScienceMap();
  } else {
    document.addEventListener('DOMContentLoaded', function () {
      bootTheme();
      mount();
      lockForgeCopy();
      bootScienceMap();
    });
  }
})();
