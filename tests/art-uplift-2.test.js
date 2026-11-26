/* ART-UPLIFT-2: finish ART-UPLIFT-1.
 * - No page keeps the old sizes="(max-width: 900px) 192px, 1400px" pattern, so phones pick the large file.
 * - Every public page in HERO_PAGES has one scenic hero whose files exist with the declared size.
 * - Utility pages privacy and contact carry the slim band, nothing else does (licence and third-party models stay as on main).
 * - Section cards on the Launch map and Examples get small destination thumbnails: each thumbnail
 *   is a crop of the destination page's own hero, lazy, decorative, inside the card's one link.
 * - The landing hero, the session-card reels and the crossfade helper stay byte-identical. */
var fs = require('fs');
var path = require('path');
var crypto = require('crypto');
var root = path.join(__dirname, '..');

function assert(cond, msg) { if (!cond) throw new Error(msg); }
function read(rel) { return fs.readFileSync(path.join(root, rel), 'utf8'); }
function sha(buf) { return crypto.createHash('sha256').update(buf).digest('hex'); }

function webpSize(buf) {
  assert(buf.slice(0, 4).toString() === 'RIFF' && buf.slice(8, 12).toString() === 'WEBP', 'not a webp');
  var kind = buf.slice(12, 16).toString();
  if (kind === 'VP8X') return { w: 1 + buf.readUIntLE(24, 3), h: 1 + buf.readUIntLE(27, 3) };
  if (kind === 'VP8 ') {
    var at = buf.indexOf(Buffer.from([0x9d, 0x01, 0x2a]));
    assert(at !== -1, 'VP8 start code missing');
    return { w: buf.readUInt16LE(at + 3) & 0x3fff, h: buf.readUInt16LE(at + 5) & 0x3fff };
  }
  if (kind === 'VP8L') {
    var b0 = buf[21], b1 = buf[22], b2 = buf[23], b3 = buf[24];
    return { w: 1 + (((b1 & 0x3f) << 8) | b0), h: 1 + (((b3 & 0x0f) << 10) | (b2 << 2) | ((b1 & 0xc0) >> 6)) };
  }
  throw new Error('unknown webp chunk ' + kind);
}
function attr(tag, name) {
  var m = tag.match(new RegExp('\\s' + name + '="([^"]*)"'));
  return m ? m[1] : null;
}
function identity(file) { return file.replace(/\.[^.]+$/, '').replace(/-\d{3,4}$/, ''); }

/* 1. The 192px sizes pattern is gone from every html file. */
var skip = { '.git': 1, node_modules: 1, target: 1, dist: 1 };
function walk(dir, out) {
  fs.readdirSync(dir, { withFileTypes: true }).forEach(function (e) {
    if (skip[e.name]) return;
    var full = path.join(dir, e.name);
    if (e.isDirectory()) return walk(full, out);
    if (/\.html$/.test(e.name)) out.push(path.relative(root, full));
  });
}
var htmlFiles = [];
walk(root, htmlFiles);
htmlFiles.forEach(function (rel) {
  assert(read(rel).indexOf('(max-width: 900px) 192px') === -1, rel + ' still uses the 192px sizes pattern');
});

/* 2. Each public page has one scenic hero with real files of the declared size. */
var HERO_PAGES = [
  'Launch-Ra-Thor.html', 'briefing.html', 'chat.html', 'constellation-week.html', 'contact.html',
  'employ.html', 'examples.html', 'go-x.html', 'licence.html', 'lineage.html', 'micro-moment.html',
  'pilot.html', 'privacy.html', 'science-watches.html', 'sovereign-shard.html', 'thanks.html',
  'third-party-models.html', 'web-forge.html'
];
/* Lead Mate council rulings 7:56 PM ET 2026-11-23: slim band only on privacy and contact. The licence
 * hero stays the full one Sherif OK'd in #42, and third-party-models keeps its live 2:1 cool crop; both match main 39acab59. */
var UTILITY = ['contact.html', 'privacy.html'];
var heroIdByPage = {};
HERO_PAGES.forEach(function (page) {
  var html = read(page);
  var figs = html.match(/<figure class="rt-art-hero[^"]*">/g) || [];
  assert(figs.length === 1, page + ' has one hero figure, got ' + figs.length);
  var start = html.indexOf(figs[0]);
  var fig = html.slice(start, html.indexOf('</figure>', start));
  var img = fig.match(/<img\b[^>]*>/)[0];
  var w = Number(attr(img, 'width')), h = Number(attr(img, 'height'));
  assert(w > 0 && h > 0, page + ' hero declares width and height');
  assert(/\d+w/.test(attr(img, 'srcset') || ''), page + ' hero has a srcset');
  assert(/100vw/.test(attr(img, 'sizes') || ''), page + ' hero sizes lets phones use the full width');
  var widest = 0, widestH = 0;
  attr(img, 'srcset').split(',').forEach(function (part) {
    var m = part.trim().match(/^\/assets\/art\/(\S+)\s+(\d+)w$/);
    assert(m, page + ' srcset entry ' + part);
    var dim = webpSize(fs.readFileSync(path.join(root, 'assets/art', m[1])));
    assert(dim.w === Number(m[2]), m[1] + ' is ' + dim.w + 'px wide, srcset says ' + m[2]);
    if (dim.w > widest) { widest = dim.w; widestH = dim.h; }
  });
  assert(widest === w && widestH === h, page + ' declared ' + w + 'x' + h + ' matches its largest file ' + widest + 'x' + widestH);
  heroIdByPage[page] = identity(attr(img, 'src').split('/').pop());
  var slim = / rt-art-hero-slim[ "]/.test(figs[0]);
  if (UTILITY.indexOf(page) !== -1) assert(slim, page + ' is a utility page and carries the slim band');
  else assert(!slim, page + ' keeps a full hero, not the slim band');
});
var pageById = {};
Object.keys(heroIdByPage).forEach(function (page) {
  var id = heroIdByPage[page];
  if (id === 'hero-legal-hall-records') return; /* the legal pair shares one scene by crop/tint (hero-uniqueness) */
  assert(!pageById[id], id + ' is the hero on both ' + pageById[id] + ' and ' + page);
  pageById[id] = page;
});

/* 2b. The legal pair is pinned to main 39acab59, byte for byte (markup and image files), and still differs by crop/tint. */
var LEGAL = {
  'licence.html': 'b1ce75f3b824c05f89316a61d50053f0e9453322bcaf38c3064169d1c78e3819',
  'third-party-models.html': '76d78598b5e75620e17535bc579cd83ddf7a60f2ff6840a8fec96af6da5cee10'
};
var legalClass = {};
Object.keys(LEGAL).forEach(function (page) {
  var html = read(page);
  var s = html.indexOf('<figure class="rt-art-hero');
  var fig = html.slice(s, html.indexOf('</figure>', s) + 9);
  assert(sha(fig) === LEGAL[page], page + ' hero markup must match main 39acab59');
  legalClass[page] = fig.match(/<figure class="([^"]*)"/)[1].split(/\s+/).sort().join(' ');
});
assert(legalClass['licence.html'] !== legalClass['third-party-models.html'], 'legal pair still differs by crop/tint');
assert(legalClass['licence.html'].indexOf('rt-art-hero-wide') === -1, 'licence keeps the full 16:9 hero');
[['assets/art/hero-legal-hall-records-768.webp', 'a3afce7078870718b463f86aa6b978b1c285349007c632f4cb269b32dbd48fb6'],
 ['assets/art/hero-legal-hall-records-1280.webp', 'fbdcd179784989390324cc2327ac72e5f4ad2574400ccbbc185b8f356abc1954']].forEach(function (p) {
  assert(sha(fs.readFileSync(path.join(root, p[0]))) === p[1], p[0] + ' must stay byte-identical');
});

/* 3. Destination thumbnails on section cards (ART-UPLIFT-3: re-cut from the new chat, shard, forge, employ and launch heroes). */
var THUMBS = {
  'Launch-Ra-Thor.html': [
    ['/chat.html', 'chat-library-counsel-eyes'],
    ['/sovereign-shard.html', 'shard-guardian-sanctum-c3'],
    ['/web-forge.html', 'forge-anvil-lattice-c3'],
    ['/employ.html', 'employ-commission-hall'],
    ['/contact.html', 'contact-comms-district']
  ],
  'examples.html': [
    ['/Launch-Ra-Thor.html', 'launch-lattice-terrace-c3'],
    ['/sovereign-shard.html', 'shard-guardian-sanctum-c3'],
    ['/web-forge.html', 'forge-anvil-lattice-c3']
  ]
};
var usedThumbs = {};
Object.keys(THUMBS).forEach(function (page) {
  var html = read(page);
  var tags = html.match(/<img class="rt-card-thumb"[^>]*>/g) || [];
  assert(tags.length === THUMBS[page].length, page + ' has ' + THUMBS[page].length + ' card thumbnails, got ' + tags.length);
  var from = 0;
  THUMBS[page].forEach(function (pair, i) {
    var tag = tags[i];
    var at = html.indexOf(tag, from);
    from = at + tag.length;
    var id = pair[1];
    assert(attr(tag, 'src') === '/assets/art/thumb-' + id + '-480.webp', page + ' thumbnail ' + i + ' is thumb-' + id);
    assert(attr(tag, 'srcset') === '/assets/art/thumb-' + id + '-480.webp 480w, /assets/art/thumb-' + id + '-720.webp 720w', page + ' thumb srcset');
    assert(attr(tag, 'width') === '720' && attr(tag, 'height') === '315', page + ' thumb declares 720x315');
    assert(attr(tag, 'loading') === 'lazy' && attr(tag, 'decoding') === 'async', page + ' thumb lazy-loads');
    assert(attr(tag, 'alt') === '', page + ' thumb is decorative; the card title names the destination');
    assert(attr(tag, 'sizes'), page + ' thumb sizes');
    [[480, 210], [720, 315]].forEach(function (d) {
      var rel = 'assets/art/thumb-' + id + '-' + d[0] + '.webp';
      var dim = webpSize(fs.readFileSync(path.join(root, rel)));
      assert(dim.w === d[0] && dim.h === d[1], rel + ' is ' + dim.w + 'x' + dim.h);
      usedThumbs[path.basename(rel)] = 1;
    });
    /* The thumbnail is a crop of the destination page's own hero. */
    var dest = pair[0].replace(/^\//, '');
    /* EXAMPLES-NO-HAMMER-1 / EXAMPLES-CANON-2 / EYES-CANON-3: a '-nh', '-c2' or '-c3' stem is the same crop, repainted (new name so cache-first sw.js refreshes it).
       The chat thumb follows the chat hero's own '-eyes' stem, so it matches without stripping. */
    assert(heroIdByPage[dest] === 'hero-' + id.replace(/-(nh|c2|c3)$/, ''), page + ' card to ' + dest + ' shows that page\'s hero (' + heroIdByPage[dest] + ')');
    /* The card stays one link: either the card is the <a>, or it has exactly one stretched link. */
    var openers = ['<a href="' + pair[0] + '" class="rt-card-uniform', '<div class="rt-card-uniform', '<section class="rt-example-card'];
    var best = -1, kind = -1;
    openers.forEach(function (o, k) { var p = html.lastIndexOf(o, at); if (p > best) { best = p; kind = k; } });
    assert(best !== -1, page + ' thumbnail ' + i + ' sits in a card');
    assert(html.slice(best, at).indexOf('</a>') === -1 && html.slice(best, at).indexOf('<a ', 1) === -1, page + ' thumbnail ' + i + ' is the first thing in its card');
    var endA = html.indexOf('</a>', at);
    var body = html.slice(best, endA + 4);
    var links = body.match(/<a\s[^>]*>/g) || [];
    assert(links.length === 1, page + ' card ' + i + ' keeps exactly one link, got ' + links.length);
    assert(attr(links[0], 'href') === pair[0], page + ' card ' + i + ' still links to ' + pair[0]);
    if (kind === 1) assert(/card-link/.test(body.slice(0, 80)) && /stretched-link/.test(links[0]), page + ' card ' + i + ' whole card is the link');
    if (kind === 2) assert(/rt-example-link/.test(links[0]), page + ' card ' + i + ' whole card is the link');
  });
});
htmlFiles.forEach(function (rel) {
  if (THUMBS[rel]) return;
  assert(read(rel).indexOf('rt-card-thumb') === -1, rel + ' has no card thumbnails in this card');
});
fs.readdirSync(path.join(root, 'assets/art')).filter(function (n) { return n.indexOf('thumb-') === 0; }).forEach(function (n) {
  assert(usedThumbs[n], n + ' is on disk and unreferenced');
});

/* 4. Locked files: landing hero, reels, crossfade helper, service worker, claim lock, chat prompt. */
var LOCKED = {
  'js/home-x-reel.js': '1b704ad9d3ec06a7656a9a0c2fd6af3fcd3b39e3df556868901d75ad44b96a4e',
  'js/home-hero.js': 'aa99b8e20dc881ed835589c6d50100f1525f163ad570e3e2ff5eeb2735cea1ed',
  'js/loop-xfade.js': '751e2c59f83d35d6916169ab6ef24dd0958ea008a96256bb2052c28bdb60b757',
  'sw.js': '7a19f245a170dbccb61e4f798c47a043eea44c99fa3bdb027314b4566708ddc6',
  'PUBLIC_CLAIM.lock.md': '0512587025bbe5a86ea2e505b40a1742792eb28c6344bd3b449716d3bc6ee22f',
  'js/chat.js': '969797f657dfb0e0b6e7455d57d9148b8373be717b70a0f318927121d1d577a0',
  'assets/art/rathor-hero-thunder-stand.mp4': 'b9016229c3f1c4bd4654d9cae90d219346ae45ae18cab12b12e06a891c9b9f23',
  'assets/art/rathor-hero-thunder-stand.jpg': 'cf747cd6ee9ca5093e60ce21b4c76071121ac89aa0a9881e37c0648eac7b4d52',
  'assets/art/reel-grok-handshake.mp4': '984e4a96f88a464d561d6279d0ca71c30d5ae77858c3f226d4b02a7d669fddf1',
  'assets/art/reel-grok-x.mp4': '893e9f8ec3cf932d10b9bd2957654395cf1e4e268aef9897de8bcd8ace42549d',
  'assets/art/reel-grok-build.mp4': '524970b0752e6050c36c6acf7cfd9608c90e745bd6521c35eb3227f0cb89622f',
  'assets/art/reel-grok-employ.mp4': '4cea95bc3619c66a751b2b6317242475e34aa26769313d95f7cad12c95d8ddba',
  'assets/art/reel-grok-build-poster.webp': 'b9c8d4b14428ed56d09bc918120ee5792a0c2e52e9c3d2cffd80b5320216de92',
  'assets/art/reel-grok-employ-poster.webp': '9c36476c826e10d52e35332aa6df068b882f71768b67d35896eb362acec46901',
  'assets/art/reel-grok-handshake-poster.webp': '0fa8f8fd65139e4dde586cc04cd86495c6ad823e6688c44163a05cf1978ed764',
  'assets/art/reel-grok-x-poster.webp': 'ca741e9a2750a7acb74bd511b91927e4ec73ff6911ab1cc93d99013fedce0093'
};
Object.keys(LOCKED).forEach(function (rel) {
  assert(sha(fs.readFileSync(path.join(root, rel))) === LOCKED[rel], rel + ' must stay byte-identical');
});
var home = read('index.html');
var vids = home.match(/<video\b[^>]*>/g) || [];
assert(vids.length === 8 && sha(vids.join('\n')) === 'd97efcfd0e4521ac3882f4b019ca4cd8ab600b9db94b08c875e2dc71a15763c5',
  'home video tags (hero, 4 session reels, 3 X reels) incl. data-loop-xfade stay byte-identical');

console.log('art-uplift-2.test.js ok (' + HERO_PAGES.length + ' heroes, ' + Object.keys(usedThumbs).length + ' thumbnail files)');
