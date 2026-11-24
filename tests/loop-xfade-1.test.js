/* LOOP-XFADE-1: runtime crossfade for opted-in looping videos.
 * Static checks: the helper exists and is small, has the reduced-motion and
 * Save-Data guards, exactly seven clips opt in (three X-reel + four session
 * reels, Lead Mate council rulings 6:55 PM ET 2026-11-23), the landing hero is
 * not opted in and stays byte-identical, and the session reels differ from
 * main only by the added attribute (mp4s, hrefs and home-x-reel.js locked). */
var fs = require('fs');
var path = require('path');
var crypto = require('crypto');
var root = path.join(__dirname, '..');

function assert(cond, msg) { if (!cond) throw new Error(msg); }
function read(rel) { return fs.readFileSync(path.join(root, rel)); }
function sha(buf) { return crypto.createHash('sha256').update(buf).digest('hex'); }

var js = read('js/loop-xfade.js').toString('utf8');
assert(js.length < 6144, 'helper stays tiny (' + js.length + ' bytes)');
assert(js.indexOf('prefers-reduced-motion: reduce') !== -1, 'reduced-motion guard');
assert(js.indexOf('saveData') !== -1, 'Save-Data guard');
assert(js.indexOf('IntersectionObserver') !== -1, 'pauses and drops the clone off-screen');
assert(js.indexOf('pointer-events:none') !== -1, 'clone never takes clicks');
assert(js.indexOf('video[loop][data-loop-xfade]') !== -1, 'opt-in only, looping videos only');
assert(!/\brequire\(|\bimport\b/.test(js), 'no dependencies');
assert(js.indexOf('visibilitychange') !== -1, 'hidden tab drops the clone');
assert(js.indexOf('readyState < 3') !== -1, 'fade only when the clone can play');
assert(js.indexOf('inset:0;width:100%;height:100%') !== -1, 'clone fills the frame like the original');
assert(js.indexOf('setTimeout(done, 1000)') !== -1, 'swap never hangs on a missed seeked');
assert((js.match(/createElement\('video'\)/g) || []).length === 1, 'one clone factory');

var html = read('index.html').toString('utf8');
var tags = html.match(/<video\b[^>]*>/g) || [];
var opted = tags.filter(function (t) { return t.indexOf('data-loop-xfade') !== -1; });
assert(opted.length === 7, 'exactly seven opted-in videos, got ' + opted.length);
opted.forEach(function (t) { assert(/\bloop\b/.test(t), 'opted-in video must loop'); });
var reelAt = html.indexOf('<section id="rt-x-reel"');
var reel = html.slice(reelAt, html.indexOf('</section>', reelAt));
assert((reel.match(/data-loop-xfade/g) || []).length === 3, 'the opt-ins are the X-reel clips');
assert(html.indexOf('<script src="/js/loop-xfade.js" defer></script>') !== -1, 'home page loads the helper with defer');

var heroAt = html.indexOf('id="rt-home-hero"');
var hero = html.slice(heroAt, html.indexOf('</figure>', heroAt));
assert(hero.indexOf('data-loop-xfade') === -1, 'landing hero has its crossfade baked into the mp4');
assert(sha(hero) === 'afb477334c4b2e925a95fad8e600ad8d63602ac194bacb6d8d36d8d051845ab3', 'hero markup byte-identical');

// Hash of each session-card block on main 90fc7a14; the only change allowed is the attribute.
var CARDS = {
  handshake: ['06b8d2e4b69a192f526b309fb96d4f04a90d8e223d4370b0d7e29143f65023a5', 'grok-cta', 'https://grok.com/share/c2hhcmQtMi1jb3B5_1f46e31d-9fe1-4982-a5e6-5a27e7517052'],
  xsession: ['da91daaeb9d6219d8ae8e1995d2cc9cebcb1cf6f14d585cf237e9af61a8d5f14', 'x-cta', '/go-x.html'],
  build: ['360a80e3333453ce69ad0682010d0d0eaa43edaada3faf8f3c5c503343ad28bb', 'vibe-cta', 'https://grok.com/share/c2hhcmQtMi1jb3B5_d08e02c6-9ceb-4e2a-b166-6dde971abcc0'],
  employ: ['dd83839a4f73ad1cab34ba3099268f11dbeb3c0fe2d3e147e46fe696a0dd1563', 'employ-cta', '/employ.html']
};
Object.keys(CARDS).forEach(function (k) {
  var a = html.lastIndexOf('<div class="card-hover', html.indexOf('data-rt-reel="' + k + '"'));
  var b = html.indexOf('</div>\n        <div class="card-hover', a);
  var e = b === -1 ? html.indexOf('Open how to employ →</a>', a) + 30 : b;
  var card = html.slice(a, e);
  var vids = card.match(/<video\b[^>]*>/g) || [];
  assert(vids.length === 1 && vids[0].indexOf(' data-loop-xfade') !== -1, k + ' session reel is opted in');
  assert(sha(card.split(' data-loop-xfade').join('')) === CARDS[k][0], k + ' session card differs from main only by data-loop-xfade');
  var ctaAt = card.indexOf('id="' + CARDS[k][1] + '"');
  var ctaTag = card.slice(card.lastIndexOf('<', ctaAt), card.indexOf('>', ctaAt) + 1);
  assert(ctaTag.indexOf('href="' + CARDS[k][2] + '"') !== -1, k + ' href unchanged');
});

var FILES = {
  'js/home-x-reel.js': '1b704ad9d3ec06a7656a9a0c2fd6af3fcd3b39e3df556868901d75ad44b96a4e',
  'js/home-hero.js': 'aa99b8e20dc881ed835589c6d50100f1525f163ad570e3e2ff5eeb2735cea1ed',
  'assets/art/rathor-hero-thunder-stand.mp4': 'b9016229c3f1c4bd4654d9cae90d219346ae45ae18cab12b12e06a891c9b9f23',
  'assets/art/reel-grok-handshake.mp4': '984e4a96f88a464d561d6279d0ca71c30d5ae77858c3f226d4b02a7d669fddf1',
  'assets/art/reel-grok-x.mp4': '893e9f8ec3cf932d10b9bd2957654395cf1e4e268aef9897de8bcd8ace42549d',
  'assets/art/reel-grok-build.mp4': '524970b0752e6050c36c6acf7cfd9608c90e745bd6521c35eb3227f0cb89622f',
  'assets/art/reel-grok-employ.mp4': '4cea95bc3619c66a751b2b6317242475e34aa26769313d95f7cad12c95d8ddba',
  'assets/art/reel-sentinel-architecture.mp4': 'c0355b1e5ad0382b0220f6e2e33b54185d6ea01e6c2a1af5b6b6a49734f47c4e',
  'assets/art/reel-tolc-heart.mp4': '38c09559c3afffda7aee42e1ebec99d827d19ae117fb2d8ed25113a425343768',
  'assets/art/reel-rathor-winged-hammer.mp4': '5061aa72c0b649cab2c8c7da0ccef053006cca27c01da09f64f21bacb88e7789'
};
Object.keys(FILES).forEach(function (f) {
  assert(sha(read(f)) === FILES[f], f + ' must stay byte-identical');
});

console.log('loop-xfade-1: ok (7 opted in, hero untouched, session reels attribute-only)');
