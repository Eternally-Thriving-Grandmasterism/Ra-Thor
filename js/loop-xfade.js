/* LOOP-XFADE-1: crossfade the end of an opted-in looping <video data-loop-xfade>
 * back into its start. One muted clone per in-view period is made near the
 * end, fades in over the loop point, then rests paused at 0 for the next loop
 * (same src, no second download). It is dropped when the clip leaves view, is
 * paused or emptied, the tab hides, or reduced motion / Save-Data turns on.
 * No video files change. Vanilla, no deps. Contact: info@Rathor.ai
 */
(function () {
  'use strict';
  var FADE = 0.7, LEAD = 2.5, all = [];
  function mq(q) { try { return window.matchMedia(q).matches; } catch (e) { return false; } }
  function off() {
    var c = navigator.connection || {};
    return mq('(prefers-reduced-motion: reduce)') || !!c.saveData;
  }
  function drop(s) {
    cancelAnimationFrame(s.raf); s.raf = 0;
    s.fading = false; s.skip = false;
    if (!s.clone) return;
    s.clone.pause();
    s.clone.removeAttribute('src');
    try { s.clone.load(); } catch (e) {}
    if (s.clone.parentNode) s.clone.parentNode.removeChild(s.clone);
    s.clone = null;
  }
  function make(v, s) {
    var c = document.createElement('video'), cs = getComputedStyle(v);
    c.muted = true; c.defaultMuted = true; c.playsInline = true;
    c.setAttribute('muted', ''); c.setAttribute('playsinline', '');
    c.setAttribute('aria-hidden', 'true'); c.tabIndex = -1;
    c.preload = 'auto';
    c.style.cssText = 'display:block;position:absolute;inset:0;width:100%;height:100%;margin:0;' +
      'pointer-events:none;opacity:0;object-fit:' + cs.objectFit + ';object-position:' + cs.objectPosition +
      ';background:' + cs.backgroundColor;
    c.src = s.src = v.currentSrc;
    v.parentNode.insertBefore(c, v.nextSibling);
    s.clone = c;
  }
  function rest(s, c) {
    if (s.clone !== c) return;
    c.pause();
    c.style.transition = 'none'; c.style.opacity = '0';
    try { c.currentTime = 0; } catch (e) {}
    s.fading = false;
  }
  function fade(v, s) {
    var c = s.clone;
    if (c.readyState < 3) { s.skip = true; return; } // not ready: plain loop, no flash
    s.fading = true;
    try { c.currentTime = 0; } catch (e) {}
    var p = c.play();
    if (p && p.catch) p.catch(function () { drop(s); });
    c.style.transition = 'opacity ' + FADE + 's linear'; c.style.opacity = '1';
    setTimeout(function () {
      if (s.clone !== c) return;
      var once = false, done = function () {
        if (once) return; once = true;
        v.removeEventListener('seeked', done);
        c.style.transition = 'opacity .3s linear'; c.style.opacity = '0';
        setTimeout(function () { rest(s, c); }, 340);
      };
      v.addEventListener('seeked', done);
      setTimeout(done, 1000); // never hang if seeked is missed
      try { v.currentTime = c.currentTime; } catch (e) { done(); }
    }, FADE * 1000 + 60);
  }
  function tick(v, s) {
    var d = v.duration, t = v.currentTime;
    if (!s.seen || off() || document.hidden || v.paused || !v.currentSrc || !isFinite(d) || d < FADE * 4 ||
        (s.clone && s.src !== v.currentSrc)) { drop(s); return; }
    if (t < d - LEAD) { s.skip = false; return; }
    if (!s.clone) make(v, s);
    if (s.fading || s.skip || s.raf || t < d - FADE - 0.4) return;
    // timeupdate is ~4 Hz; watch the last stretch per frame so the fade starts on time
    var step = function () {
      s.raf = 0;
      if (!s.clone || s.fading || s.skip || v.paused) return;
      if (v.currentTime >= v.duration - FADE) fade(v, s);
      else if (v.currentTime >= v.duration - LEAD) s.raf = requestAnimationFrame(step);
    };
    step();
  }
  function wire(v) {
    if (v.getAttribute('data-loop-xfade-on')) return;
    v.setAttribute('data-loop-xfade-on', '1');
    var s = { seen: false, clone: null, fading: false, skip: false, held: false, raf: 0, src: '' };
    all.push(s);
    v.addEventListener('timeupdate', function () { tick(v, s); });
    v.addEventListener('emptied', function () { drop(s); });
    v.addEventListener('pause', function () { drop(s); });
    if (typeof IntersectionObserver !== 'function') { s.seen = true; return; }
    new IntersectionObserver(function (es) {
      s.seen = es[es.length - 1].isIntersecting;
      if (!s.seen) {
        drop(s);
        if (!v.paused) { s.held = true; v.pause(); }
      } else if (s.held) {
        s.held = false;
        var p = v.play();
        if (p && p.catch) p.catch(function () {});
      }
    }, { rootMargin: '200px 0px' }).observe(v);
  }
  function boot() {
    var list = document.querySelectorAll('video[loop][data-loop-xfade]');
    for (var i = 0; i < list.length; i++) wire(list[i]);
    document.addEventListener('visibilitychange', function () {
      if (document.hidden) for (var j = 0; j < all.length; j++) drop(all[j]);
    });
  }
  if (typeof document === 'undefined' || !document.querySelectorAll) return;
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
})();
