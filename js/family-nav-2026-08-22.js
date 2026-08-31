/* family-nav-2026-08-22.js
   Shared public-surface network for rathor.ai
   Workspace 14.15.6
   Contact: info@Rathor.ai
   2026-08-23: Install chip + coherent family labels
   2026-08-23b: Professional site footer contract · no pill-bar footer
   2026-08-23c: Celestial sun/moon orb + compact install glyph
   2026-08-23d: Skip link is fixed+clip — never left:-999px (standalone RTL)
   2026-08-23e: Uniform gold bar — incomplete local navs no longer suppress;
                retire competing family navs; Chat + Forge + Shard share one chrome
   2026-08-31: Public speech lock — no APTD badge in fallback footer
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
    if (document.getElementById('rt-family-nav')) return true;
    var scope = document.querySelector('header') || document.body;
    if (!scope) return false;
    var found = {};
    var anchors = scope.querySelectorAll('a');
    for (var i = 0; i < anchors.length; i++) {
      var t = (anchors[i].textContent || '').replace(/\s+/g, ' ').trim();
      if (t) found[t] = true;
    }
    return !!(found.Home && found.Chat && found.Launch && found.Moments && found.Shard && found.Forge && found.Contact && found.Privacy);
  }

  function retireLocalFamilyNavs() {
    var nodes = document.querySelectorAll('nav');
    for (var i = 0; i < nodes.length; i++) {
      var n = nodes[i];
      if (n.id === 'rt-family-nav' || n.id === 'rt-family-footer') continue;
      var label = (n.getAttribute('aria-label') || '').toLowerCase();
      if (label === 'ra-thor family' || label.indexOf('ra-thor family') !== -1) {
        if (n.parentNode) n.parentNode.removeChild(n);
      }
    }
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

    var tools = document.createElement('span');
    tools.className = 'rt-nav-tools';
    tools.setAttribute('aria-label', 'Site tools');

    if (!isStandalone()) {
      var install = document.createElement('button');
      install.type = 'button';
      install.className = 'rt-install-orb';
      install.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M12 3v12m0 0-4-4m4 4 4-4M5 19h14" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>';
      install.setAttribute('aria-label', 'Install Ra-Thor on this device');
      install.setAttribute('data-rt-say-skip', '1');
      install.title = 'Install on this device';
      install.addEventListener('click', function () {
        if (typeof window.rathorTriggerPWAInstall === 'function') {
          window.rathorTriggerPWAInstall();
        } else if (typeof window.rathorSay === 'function') {
          window.rathorSay({
            title: 'Install module loading',
            body: 'Give it a second, then tap Install again.',
            tone: 'hold',
            ms: 3000
          });
        }
      });
      tools.appendChild(install);
    }

    var theme = document.createElement('button');
    theme.type = 'button';
    theme.id = kind === 'top' ? 'rt-theme-toggle' : 'rt-theme-toggle-foot';
    theme.className = 'rt-theme-orb';
    theme.setAttribute('data-rt-theme-toggle', '1');
    theme.setAttribute('data-rt-say-skip', '1');
    theme.setAttribute('aria-label', 'Switch to light mode');
    theme.title = 'Day — switch to light';
    theme.innerHTML = '<span class="rt-celestial-stack" aria-hidden="true"></span>';
    theme.addEventListener('click', function () {
      if (typeof window.rathorToggleTheme === 'function') window.rathorToggleTheme();
    });
    tools.appendChild(theme);
    nav.appendChild(tools);
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
            '<p class="rt-legal">v14.15.6 · AG-SML v1.1 · TOLC 8 · Capable · Bounded · Corrigible</p>' +
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
          '<div>© 2026 Sherif Samy Botros — sole steward of Autonomicity Games Inc. & AlphaProMega Air Foundation. TOLC 8 · independent of xAI.</div>' +
          '<a href="mailto:info@Rathor.ai">info@Rathor.ai</a>' +
        '</div>' +
      '</div>';
    return wrap;
  }

  var SKIP_HIDE =
    'position:fixed;top:0;left:0;inset-inline-start:0;inset-inline-end:auto;' +
    'width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;' +
    'clip:rect(0,0,0,0);clip-path:inset(50%);white-space:nowrap;border:0;' +
    'z-index:60;background:transparent;color:transparent;';
  var SKIP_SHOW =
    'position:fixed;top:0.5rem;inset-inline-start:0.5rem;left:auto;right:auto;' +
    'width:auto;height:auto;padding:0.4rem 0.7rem;margin:0;overflow:visible;' +
    'clip:auto;clip-path:none;white-space:nowrap;z-index:80;' +
    'background:#fbbf24;color:#000;border-radius:0.35rem;' +
    'font:600 12px/1.2 system-ui,sans-serif;';

  function hideSkip(a) {
    if (!a) return;
    a.style.cssText = SKIP_HIDE;
  }

  function clipViewport() {
    var root = document.documentElement;
    if (root) {
      root.style.overflowX = 'clip';
      root.style.maxWidth = '100%';
    }
    if (document.body) {
      document.body.style.overflowX = 'clip';
      document.body.style.maxWidth = '100%';
    }
  }

  function skipLink() {
    var existing = document.getElementById('rt-skip-family');
    if (existing) {
      hideSkip(existing);
      return;
    }
    var a = document.createElement('a');
    a.id = 'rt-skip-family';
    a.href = '#rt-family-main';
    a.textContent = 'Skip family navigation';
    hideSkip(a);
    a.addEventListener('focus', function () { a.style.cssText = SKIP_SHOW; });
    a.addEventListener('blur', function () { hideSkip(a); });
    document.body.insertBefore(a, document.body.firstChild);
    if (!document.getElementById('rt-family-main')) {
      var main = document.querySelector('main') || document.body.children[1] || document.body;
      if (main && main !== a) main.id = main.id || 'rt-family-main';
    }
  }

  function mount() {
    if (document.body && document.body.getAttribute('data-rt-family') === 'off') return;
    clipViewport();
    skipLink();
    retireLocalFamilyNavs();
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
    if (!window.rathorSay && !document.querySelector('script[src*="rathor-feedback"]')) {
      var f = document.createElement('script');
      f.src = '/js/rathor-feedback.js';
      (document.head || document.documentElement).appendChild(f);
    }
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
