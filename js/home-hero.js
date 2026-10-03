/* Home hero — still image first. Attach the muted mp4 only when motion,
 * network, and width allow. Without this script the picture remains.
 * window.rtMutedClip is the same gate the home X reel uses.
 * Contact: info@Rathor.ai
 */
(function () {
  'use strict';

  var MP4 = '/assets/art/hero-home-rathor.mp4';
  var POSTER = '/assets/art/hero-home-rathor-poster-784.webp';

  function heroVideoAllowed(env) {
    env = env || {};
    if (env.reducedMotion) return false;
    if (env.saveData) return false;
    var kind = env.effectiveType || '';
    if (kind === 'slow-2g' || kind === '2g' || kind === '3g') return false;
    if (env.narrow) return false;
    return true;
  }

  function readEnv() {
    var reduced = false;
    var narrow = false;
    try { reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches; } catch (e1) {}
    try { narrow = window.matchMedia('(max-width: 640px)').matches; } catch (e2) {}
    var conn = navigator.connection || navigator.mozConnection || navigator.webkitConnection || null;
    return {
      reducedMotion: reduced,
      narrow: narrow,
      saveData: !!(conn && conn.saveData),
      effectiveType: conn && conn.effectiveType ? String(conn.effectiveType) : ''
    };
  }

  function silence(video) {
    video.defaultMuted = true;
    video.muted = true;
    video.volume = 0;
  }

  function releaseMutedClip(video) {
    if (!video) return;
    video.pause();
    silence(video);
    video.removeAttribute('src');
    video.removeAttribute('poster');
    var nodes = video.querySelectorAll('source');
    for (var i = 0; i < nodes.length; i++) nodes[i].parentNode.removeChild(nodes[i]);
    video.removeAttribute('data-rt-live');
    try { video.load(); } catch (e3) {}
  }

  function bindMutedClip(video, src, poster) {
    if (!video || video.getAttribute('data-rt-live') === '1') return;
    silence(video);
    video.setAttribute('muted', '');
    video.autoplay = true;
    video.loop = true;
    video.playsInline = true;
    video.preload = 'none';
    video.setAttribute('aria-hidden', 'true');
    video.controls = false;
    if (poster) video.poster = poster;
    var source = document.createElement('source');
    source.type = 'video/mp4';
    source.src = src;
    video.appendChild(source);
    video.setAttribute('data-rt-live', '1');
    try { video.load(); } catch (e4) {}
    var pending = video.play();
    if (pending && typeof pending.catch === 'function') pending.catch(function () {});
  }

  function pauseMutedClip(video) {
    if (!video) return;
    video.pause();
    silence(video);
  }

  function watchEnv(onChange) {
    function watch(mq) {
      if (!mq) return;
      if (typeof mq.addEventListener === 'function') mq.addEventListener('change', onChange);
      else if (typeof mq.addListener === 'function') mq.addListener(onChange);
    }
    try { watch(window.matchMedia('(prefers-reduced-motion: reduce)')); } catch (e5) {}
    try { watch(window.matchMedia('(max-width: 640px)')); } catch (e6) {}
    var conn = navigator.connection || navigator.mozConnection || navigator.webkitConnection;
    if (conn && typeof conn.addEventListener === 'function') conn.addEventListener('change', onChange);
  }

  var rtMutedClip = {
    heroVideoAllowed: heroVideoAllowed,
    readEnv: readEnv,
    silence: silence,
    bind: bindMutedClip,
    release: releaseMutedClip,
    pause: pauseMutedClip,
    watchEnv: watchEnv
  };
  if (typeof window !== 'undefined') {
    window.rtHomeHeroVideoAllowed = heroVideoAllowed;
    window.rtMutedClip = rtMutedClip;
  }
  if (typeof module !== 'undefined' && module.exports) module.exports = rtMutedClip;

  function boot() {
    var figure = document.getElementById('rt-home-hero');
    if (!figure) return;
    var video = figure.querySelector('video');
    if (!video) return;

    function attach() {
      if (video.getAttribute('data-rt-live') === '1') return;
      if (!heroVideoAllowed(readEnv())) return;
      var still = figure.querySelector('img');
      var poster = (still && still.currentSrc) ? still.currentSrc : POSTER;
      bindMutedClip(video, MP4, poster);
    }

    function sync() {
      if (heroVideoAllowed(readEnv())) attach();
      else releaseMutedClip(video);
    }

    video.addEventListener('volumechange', function () {
      if (!video.muted || video.volume !== 0) silence(video);
    });

    watchEnv(sync);

    if (!heroVideoAllowed(readEnv())) {
      releaseMutedClip(video);
      return;
    }
    function afterPaint() {
      if (window.requestAnimationFrame) {
        window.requestAnimationFrame(function () { window.requestAnimationFrame(sync); });
      } else {
        sync();
      }
    }
    afterPaint();
  }

  if (typeof document === 'undefined' || !document.getElementById) return;
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
})();
