/* RATHOR-VIDEO-1: three steward clips under the home intro.
 * HOME-CLIPS-2: two more clips on the Ra-Thor + Grok and Build with Grok cards.
 * HOME-CLIPS-3: two more clips on the Ra-Thor on X and How to employ cards.
 * Stills are the default. js/home-x-reel.js attaches muted mp4s
 * through the shared home-hero gate, and only when a frame is near view.
 */
var fs = require('fs');
var path = require('path');
var root = path.join(__dirname, '..');

function assert(cond, msg) {
  if (!cond) throw new Error(msg);
}

function read(rel) {
  return fs.readFileSync(path.join(root, rel), 'utf8');
}

var files = {
  'assets/art/reel-sentinel-architecture.mp4': 824868,
  'assets/art/reel-sentinel-architecture-poster.webp': 34554,
  'assets/art/reel-tolc-heart.mp4': 1472913,
  'assets/art/reel-tolc-heart-poster.webp': 53848,
  'assets/art/reel-rathor-winged-hammer.mp4': 1397223,
  'assets/art/reel-rathor-winged-hammer-poster.webp': 58124,
  'assets/art/reel-grok-handshake.mp4': 8037507,
  'assets/art/reel-grok-handshake-poster.webp': 161624,
  'assets/art/reel-grok-build.mp4': 8896114,
  'assets/art/reel-grok-build-poster.webp': 157062,
  'assets/art/reel-grok-x.mp4': 6769021,
  'assets/art/reel-grok-x-poster.webp': 97898,
  'assets/art/reel-grok-employ.mp4': 8710848,
  'assets/art/reel-grok-employ-poster.webp': 113738
};
Object.keys(files).forEach(function (rel) {
  var full = path.join(root, rel);
  assert(fs.existsSync(full), rel + ' must exist');
  assert(fs.statSync(full).size === files[rel], rel + ' must stay the supplied ' + files[rel] + '-byte file');
});

var html = read('index.html');
assert(html.indexOf('reel-sentinel-architecture.mp4') === -1, 'index must not embed the sentinel mp4');
assert(html.indexOf('reel-tolc-heart.mp4') === -1, 'index must not embed the tolc mp4');
assert(html.indexOf('reel-rathor-winged-hammer.mp4') === -1, 'index must not embed the hammer mp4');
assert(html.indexOf('reel-grok-handshake.mp4') === -1, 'index must not embed the handshake mp4');
assert(html.indexOf('reel-grok-build.mp4') === -1, 'index must not embed the build mp4');
assert(html.indexOf('reel-grok-x.mp4') === -1, 'index must not embed the x mp4');
assert(html.indexOf('reel-grok-employ.mp4') === -1, 'index must not embed the employ mp4');
assert(html.indexOf('video.twimg.com') === -1, 'index must not leave a video.twimg.com src');
assert(html.indexOf('2078966782883135582') === -1, 'hammer post must not be linked; its text is not repeated');
assert(html.indexOf('/js/home-x-reel.js') !== -1, 'index must load home-x-reel.js');
assert(html.indexOf('/js/home-hero.js') < html.indexOf('/js/home-x-reel.js'), 'hero gate script must load before the reel');

var heroStart = html.indexOf('id="rt-home-hero"');
var heroEnd = html.indexOf('</figure>', heroStart);
assert(heroStart !== -1 && heroEnd > heroStart, 'home hero figure must stay');
var hero = html.slice(heroStart, heroEnd);
assert(hero.indexOf('reel-') === -1, 'home hero figure must not gain the reel');
assert(hero.indexOf('hero-home-rathor-poster-784.webp') !== -1, 'home hero still must stay');

var reelStart = html.indexOf('id="rt-x-reel"');
var reelEnd = html.indexOf('</section>', reelStart);
assert(reelStart !== -1 && reelEnd > reelStart, 'reel section must exist');
assert(html.indexOf('id="steward-line"') < reelStart && reelStart < html.indexOf('id="lang-selector"'), 'reel sits under the intro and above the language row');
var reel = html.slice(reelStart, reelEnd);
assert(reel.indexOf('data-i18n') === -1, 'reel copy stays verbatim and is not an i18n key');
assert(reel.indexOf('From <a href="https://x.com/AlphaProMega" rel="noopener" target="_blank">@AlphaProMega</a> on X') !== -1, 'heading stays plain');
assert(reel.indexOf('Eternal Sentinel Architecture v8.0 visualized') !== -1, 'sentinel caption stays verbatim');
assert(reel.indexOf('https://x.com/AlphaProMega/status/2001770410904317989') !== -1, 'sentinel caption links to its post');
assert(reel.indexOf('for all creations and creatures, to thrive with positive emotions forever and ever') !== -1, 'tolc caption stays verbatim');
assert(reel.indexOf('https://x.com/AlphaProMega/status/2057902030614601883') !== -1, 'tolc caption links to its post');
assert(reel.indexOf('rel="noopener"') !== -1 && reel.indexOf('target="_blank"') !== -1, 'caption links open safely');

var alts = [
  ['464', '664', 'A luminous blue-white humanoid figure stands between a bright sun and a spiral galaxy, inside glowing circles and geometric marks.'],
  ['640', '952', 'A blue lightning-lined robot figure holds out an open hand under a glowing pink heart with the word TOLC.'],
  ['960', '644', 'A dark hammer with fiery wings and lightning sits in front of a glowing ring, above fiery title text reading RA-THOR.']
];
alts.forEach(function (row) {
  assert(reel.indexOf('width="' + row[0] + '"') !== -1 && reel.indexOf('height="' + row[1] + '"') !== -1, row[0] + ' poster must set width and height');
  assert(reel.indexOf('alt="' + row[2] + '"') !== -1, 'alt must describe the picture');
});
assert((reel.match(/loading="lazy"/g) || []).length === 3, 'each poster img is lazy');
assert((reel.match(/<figcaption\b/g) || []).length === 2, 'only the two captioned posts have figcaptions');
assert(reel.indexOf('data-rt-reel="hammer"') !== -1, 'hammer frame must be present');
var hammerAt = reel.indexOf('data-rt-reel="hammer"');
var hammerFig = reel.slice(hammerAt, reel.indexOf('</figure>', hammerAt));
assert(hammerFig.indexOf('<figcaption') === -1, 'hammer has no caption');

var videos = reel.match(/<video\b[^>]*>/g);
assert(videos && videos.length === 3, 'three video shells');
videos.forEach(function (tag) {
  assert(tag.indexOf('muted') !== -1, 'video must be muted');
  assert(tag.indexOf('autoplay') !== -1, 'video must autoplay');
  assert(tag.indexOf('loop') !== -1, 'video must loop');
  assert(tag.indexOf('playsinline') !== -1, 'video must be playsinline');
  assert(tag.indexOf('preload="none"') !== -1, 'video preload must be none');
  assert(tag.indexOf('aria-hidden="true"') !== -1, 'video duplicates the alt and must be aria-hidden');
  assert(tag.indexOf('controls') === -1, 'video must not show controls');
});
assert(reel.indexOf('<source') === -1 && reel.indexOf('video/mp4') === -1, 'static reel must not include an mp4 source');
assert(html.indexOf('@media (prefers-reduced-motion: reduce)') !== -1 && html.indexOf('.rt-x-frame video { display: none !important; }') !== -1, 'reduced motion hides reel video');

var js = read('js/home-x-reel.js');
assert(js.indexOf('reel-sentinel-architecture.mp4') !== -1, 'script must name the sentinel mp4');
assert(js.indexOf('reel-tolc-heart.mp4') !== -1, 'script must name the tolc mp4');
assert(js.indexOf('reel-rathor-winged-hammer.mp4') !== -1, 'script must name the hammer mp4');
assert(js.indexOf('reel-grok-handshake.mp4') !== -1, 'script must name the handshake mp4');
assert(js.indexOf('reel-grok-build.mp4') !== -1, 'script must name the build mp4');
assert(js.indexOf('reel-grok-x.mp4') !== -1, 'script must name the x mp4');
assert(js.indexOf('reel-grok-employ.mp4') !== -1, 'script must name the employ mp4');
assert(js.indexOf('video.twimg.com') === -1, 'script must not leave a video.twimg.com src');
assert(js.indexOf("document.querySelectorAll('[data-rt-reel]')") !== -1, 'card clips use the same reel wiring');
assert(js.indexOf('heroVideoAllowed') !== -1, 'reel must use the shared hero gate');
assert(js.indexOf('slow-2g') === -1 && js.indexOf('saveData') === -1, 'reel must not copy the gate conditions');
assert(js.indexOf('IntersectionObserver') !== -1, 'reel must watch viewport nearness');
assert(js.indexOf("rootMargin: '200px 0px'") !== -1, 'near view uses a 200px root margin');
assert(js.indexOf('api.pause(video)') !== -1, 'leaving view pauses');
assert(js.indexOf('api.release(video)') !== -1, 'a closed gate drops the source');
assert(js.indexOf('api.watchEnv(resync)') !== -1, 'reel follows the hero env watch');
assert(js.indexOf('still.currentSrc') !== -1, 'poster must reuse the still that already loaded');

var heroApi = require(path.join(root, 'js/home-hero.js'));
var reelApi = require(path.join(root, 'js/home-x-reel.js'));
function deny(env, near, label) {
  assert(reelApi.reelClipShouldPlay(env, near) === false, label);
}
function allow(env, near, label) {
  assert(reelApi.reelClipShouldPlay(env, near) === true, label);
}
var wide = { reducedMotion: false, narrow: false, saveData: false, effectiveType: '4g' };
deny(Object.assign({}, wide, { reducedMotion: true }), true, 'reduced motion blocks video');
deny(Object.assign({}, wide, { saveData: true }), true, 'saveData blocks video');
deny(Object.assign({}, wide, { effectiveType: 'slow-2g' }), true, 'slow-2g blocks video');
deny(Object.assign({}, wide, { effectiveType: '2g' }), true, '2g blocks video');
deny(Object.assign({}, wide, { effectiveType: '3g' }), true, '3g blocks video');
deny(Object.assign({}, wide, { narrow: true }), true, 'narrow viewport blocks video');
deny(wide, false, 'offscreen blocks video even on a fast network');
allow(wide, true, '4g wide and near allows video');
allow(Object.assign({}, wide, { effectiveType: '' }), true, 'unknown network and near allows video');
assert(heroApi.pause && heroApi.release && heroApi.bind && heroApi.watchEnv, 'shared clip helpers stay on the hero module');

['sw.js', 'public/sw.js', 'offline.html'].forEach(function (rel) {
  var text = read(rel);
  Object.keys(files).forEach(function (media) {
    assert(text.indexOf(path.basename(media)) === -1, rel + ' must not precache ' + media);
  });
  assert(text.indexOf('home-x-reel') === -1, rel + ' must not reference the reel script');
});

['contact.html', 'micro-moment.html', 'pilot.html', 'Launch-Ra-Thor.html', 'constellation-week.html'].forEach(function (rel) {
  var page = read(rel);
  assert(page.indexOf('rt-x-reel') === -1 && page.indexOf('reel-sentinel-architecture') === -1, rel + ' must not take the reel');
});

var headers = read('_headers');
assert(headers.indexOf('media-src') === -1, '_headers must not send a media-src that would block same-origin clips');
assert(html.indexOf('Content-Security-Policy') === -1, 'index must not add a CSP meta that blocks self media');

function cardSlice(startId, endId) {
  var start = html.indexOf('id="' + startId + '"');
  var end = html.indexOf('id="' + endId + '"');
  if (end <= start) end = html.indexOf(endId, start + 1);
  assert(start !== -1 && end > start, startId + ' card slice');
  var open = html.lastIndexOf('<div class="card-hover', start);
  assert(open !== -1 && open < start, startId + ' card shell');
  return html.slice(open, end);
}

function assertClip(slice, key, poster, alt, ctaId, sessionHref, postUrl, newTab) {
  assert(slice.indexOf('rt-home-clip-card') !== -1, key + ' card keeps the clip shell');
  assert(slice.indexOf('data-rt-reel="' + key + '"') !== -1, key + ' frame must be present');
  assert(slice.indexOf('data-rt-reel="' + key + '"') < slice.indexOf('id="' + ctaId + '"'), key + ' clip sits above the button');
  assert(slice.indexOf(poster) !== -1, key + ' poster must be on the card');
  assert(slice.indexOf('width="960"') !== -1 && slice.indexOf('height="644"') !== -1, key + ' poster must set width and height');
  assert(slice.indexOf('alt="' + alt + '"') !== -1, key + ' alt must describe the picture');
  assert(slice.indexOf('loading="lazy"') !== -1, key + ' poster img is lazy');
  assert(slice.indexOf('Draft visual. Not a product.') === -1, key + ' has no disclaimer caption');
  assert(slice.indexOf('<figcaption') === -1, key + ' has no caption');
  assert(slice.indexOf('Infinitely Winning') === -1, key + ' must not say Infinitely Winning');
  assert(slice.indexOf('AGSi') === -1, key + ' must not state AGSi');
  assert(slice.indexOf('data-i18n="') !== -1, key + ' card copy keys stay');
  var videos = slice.match(/<video\b[^>]*>/g);
  assert(videos && videos.length === 1, key + ' has one video shell');
  assert(videos[0].indexOf('muted') !== -1, key + ' video must be muted');
  assert(videos[0].indexOf('autoplay') !== -1, key + ' video must autoplay');
  assert(videos[0].indexOf('loop') !== -1, key + ' video must loop');
  assert(videos[0].indexOf('playsinline') !== -1, key + ' video must be playsinline');
  assert(videos[0].indexOf('preload="none"') !== -1, key + ' video preload must be none');
  assert(videos[0].indexOf('aria-hidden="true"') !== -1, key + ' video duplicates the alt and must be aria-hidden');
  assert(videos[0].indexOf('controls') === -1, key + ' video must not show controls');
  assert(videos[0].indexOf('src=') === -1, key + ' video shell must not embed a src');
  var ctaAt = slice.indexOf('id="' + ctaId + '"');
  var ctaTag = slice.slice(slice.lastIndexOf('<', ctaAt), slice.indexOf('>', ctaAt) + 1);
  assert(ctaTag.indexOf('<a ') === 0, key + ' session control stays a link');
  assert(ctaTag.indexOf('href="' + sessionHref + '"') !== -1, key + ' session href stays on the button');
  if (newTab === false) {
    assert(ctaTag.indexOf('target=') === -1, key + ' button stays a same-page link');
  } else {
    assert(ctaTag.indexOf('target="_blank"') !== -1 && ctaTag.indexOf('rel="noopener"') !== -1, key + ' button opens in a new tab');
  }
}

var grokCard = cardSlice('grok-title', 'x-title');
assertClip(
  grokCard,
  'handshake',
  'reel-grok-handshake-poster.webp',
  'A gold-armored warrior with a winged helmet and an Eye-of-Horus halo holds a hammer and clasps hands with a dark figure traced in stars, inside a ring of fire in a hall of glowing windows.',
  'grok-cta',
  'https://grok.com/share/c2hhcmQtMi1jb3B5_1f46e31d-9fe1-4982-a5e6-5a27e7517052'
);
var buildCard = cardSlice('vibe-title', 'employ-title');
assertClip(
  buildCard,
  'build',
  'reel-grok-build-poster.webp',
  'Two lavender-skinned Quellorians in white-and-gold robes build a pearl-white craft held in a gantry, one fitting its engine and one sorting tools at a workbench, in a sunlit workshop above a turquoise bay.',
  'vibe-cta',
  'https://grok.com/share/c2hhcmQtMi1jb3B5_d08e02c6-9ceb-4e2a-b166-6dde971abcc0'
);
var xCard = cardSlice('x-title', 'vibe-title');
assertClip(
  xCard,
  'xsession',
  'reel-grok-x-poster.webp',
  'A gold-armored warrior with a winged helmet and an Eye-of-Horus halo holds a hammer beside a blue figure traced in stars, in front of a tall window filled with a glowing X.',
  'x-cta',
  '/go-x.html',
  'https://x.com/AlphaProMega/status/2107689413517979810',
  true
);
var employCard = cardSlice('employ-title', 'homeLaunchMap');
assertClip(
  employCard,
  'employ',
  'reel-grok-employ-poster.webp',
  'A gold-armored warrior with a winged helmet and an Eye-of-Horus halo holds a hammer beside a blue figure traced in stars, both facing forward in front of three equal glowing doorways.',
  'employ-cta',
  '/employ.html',
  'https://x.com/AlphaProMega/status/2107689413517979810',
  false
);
assert(html.indexOf('<a href="/go-x.html" target="_blank" rel="noopener" class="card-hover') === -1, 'X card is not one link');
assert(html.indexOf('<a href="/employ.html" class="card-hover') === -1, 'employ card is not one link');
assert(grokCard.indexOf('2107689413517979810') === -1 && grokCard.indexOf('data-rt-reel="xsession"') === -1, 'grok card stays the handshake clip');
assert(buildCard.indexOf('2107689413517979810') === -1 && buildCard.indexOf('data-rt-reel="employ"') === -1, 'build card stays the hammer clip');
assert(reelApi.CLIPS.xsession === '/assets/art/reel-grok-x.mp4', 'xsession key points at the local mp4');
assert(reelApi.CLIPS.employ === '/assets/art/reel-grok-employ.mp4', 'employ key points at the local mp4');
assert(reel.indexOf('reel-grok-') === -1 && reel.indexOf('2107582671530475617') === -1 && reel.indexOf('2107689413517979810') === -1, 'intro reel does not take the card clips');
assert(html.indexOf('Draft visual. Not a product.') === -1, 'home page has no disclaimer caption');

console.log('home-x-reel.test.js ok');
