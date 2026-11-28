/* ART-UPLIFT-3: new Ra-Thor heroes where the figure fits the page, theme redo for privacy and thanks,
 * a new pilot hero and a canon redo of the examples hammer. Council ruling (4), 7:56 PM ET 2026-11-23.
 * - Each changed page uses its new hero (unique identity, files exist, declared size matches).
 * - The replaced heroes and thumbnails are gone from disk and from every non-test file.
 * - Everything this card must not touch stays byte-identical (sha256).
 * LICENCE-SELF-SERVE-1: index.html (footer privacy line) and licence.html (small-organisation prices, LICENSE §2 quote) re-pinned; hero markup unchanged.
 * SUPPORTER-PASS-1: index.html re-pinned for the new #supporter-pass section (hero markup unchanged), then again for the AGi Discord server name (v5).
 * GOLIVE-PREP-1: index.html and licence.html re-pinned for the GOLIVE-SWAP marker comments (no visible change). */
var fs = require('fs');
var path = require('path');
var crypto = require('crypto');
var root = path.join(__dirname, '..');
function assert(cond, msg) { if (!cond) throw new Error(msg); }
function read(rel) { return fs.readFileSync(path.join(root, rel), 'utf8'); }
function sha(buf) { return crypto.createHash('sha256').update(buf).digest('hex'); }
function figure(html) {
  var s = html.indexOf('<figure class="rt-art-hero');
  assert(s !== -1, 'hero figure missing');
  return html.slice(s, html.indexOf('</figure>', s) + 9);
}
var NEW = {
  "Launch-Ra-Thor.html": "hero-launch-lattice-terrace",
  "briefing.html": "hero-briefing-bridge-landing",
  "chat.html": "hero-chat-library-counsel-eyes2",
  "employ.html": "hero-employ-commission-hall",
  "web-forge.html": "hero-forge-anvil-lattice",
  "sovereign-shard.html": "hero-shard-guardian-sanctum",
  "privacy.html": "hero-privacy-sealed-vault",
  "thanks.html": "hero-thanks-message-relay",
  "pilot.html": "hero-pilot-scoped-workbench",
  "examples.html": "hero-examples-workbench-c3"
};
var seen = {};
Object.keys(NEW).forEach(function (page) {
  var fig = figure(read(page));
  var src = fig.match(/\bsrc="\/assets\/art\/([^"]+)"/)[1];
  assert(src.indexOf(NEW[page] + '-') === 0, page + ' uses ' + NEW[page] + ', got ' + src);
  assert(!seen[NEW[page]], NEW[page] + ' is used twice');
  seen[NEW[page]] = page;
  fig.match(/\bsrcset="([^"]*)"/)[1].split(',').forEach(function (part) {
    var f = part.trim().split(/\s+/)[0].replace('/assets/art/', '');
    assert(f.indexOf(NEW[page] + '-') === 0, page + ' srcset only lists its new hero');
    assert(fs.existsSync(path.join(root, 'assets/art', f)), f + ' exists');
  });
  assert(/fetchpriority="high"/.test(fig) && !/loading="lazy"/.test(fig), page + ' lead hero fetches first');
  assert(/\balt="[^"]{40,}"/.test(fig), page + ' has a descriptive English alt');
});
assert(/rt-art-hero-slim/.test(figure(read('privacy.html'))), 'privacy keeps the slim band');
var OLD = ["hero-launch-shuttle-port", "hero-briefing-expedition-bridge", "hero-chat-research-district", "hero-employ-briefing-room", "hero-forge-workshop", "hero-shard-crystal-city", "hero-privacy-discovery-gallery", "hero-thanks-cable-car", "hero-pilot-sunrise-crew", "hero-examples-workbench", "thumb-launch-shuttle-port", "thumb-chat-research-district", "thumb-employ-briefing-room", "thumb-forge-workshop", "thumb-shard-crystal-city"];
var skip = { '.git': 1, node_modules: 1, target: 1, dist: 1, tests: 1 };
function walk(dir, out) {
  fs.readdirSync(dir, { withFileTypes: true }).forEach(function (e) {
    if (skip[e.name]) return;
    var full = path.join(dir, e.name);
    if (e.isDirectory()) return walk(full, out);
    if (/\.(html|js|css|json|md|txt|xml|webmanifest)$/.test(e.name)) out.push(full);
  });
}
var files = [];
walk(root, files);
var art = fs.readdirSync(path.join(root, 'assets/art'));
OLD.forEach(function (stem) {
  assert(!art.some(function (f) { return f.indexOf(stem + '-') === 0 && /-\d+\.webp$/.test(f) && f.replace(/-\d+\.webp$/, '') === stem; }), stem + ' files are removed');
  var re = new RegExp(stem + '-\\d');
  files.forEach(function (f) { assert(!re.test(fs.readFileSync(f, 'utf8')), path.relative(root, f) + ' still names ' + stem); });
});
var PINS = {
  "index.html": "217f1e373455a504a99322b166aeef3db170aadff2a663c5d6c1906fe2faddf5",
  "licence.html": "ab4d3c4e8c994a16147b62431ffa1d653409988986d8da193471355f03e64189",
  "third-party-models.html": "653c32d1a932f3478ce7655e1232a530f5518dde10a741fd272dbb0783be4931",
  "js/loop-xfade.js": "751e2c59f83d35d6916169ab6ef24dd0958ea008a96256bb2052c28bdb60b757",
  "js/home-x-reel.js": "1b704ad9d3ec06a7656a9a0c2fd6af3fcd3b39e3df556868901d75ad44b96a4e",
  "js/home-hero.js": "aa99b8e20dc881ed835589c6d50100f1525f163ad570e3e2ff5eeb2735cea1ed",
  "sw.js": "7a19f245a170dbccb61e4f798c47a043eea44c99fa3bdb027314b4566708ddc6",
  "PUBLIC_CLAIM.lock.md": "0512587025bbe5a86ea2e505b40a1742792eb28c6344bd3b449716d3bc6ee22f",
  "js/chat.js": "969797f657dfb0e0b6e7455d57d9148b8373be717b70a0f318927121d1d577a0",
  "assets/art/reel-grok-build-poster.webp": "b9c8d4b14428ed56d09bc918120ee5792a0c2e52e9c3d2cffd80b5320216de92",
  "assets/art/reel-grok-build.mp4": "524970b0752e6050c36c6acf7cfd9608c90e745bd6521c35eb3227f0cb89622f",
  "assets/art/reel-grok-employ-poster.webp": "9c36476c826e10d52e35332aa6df068b882f71768b67d35896eb362acec46901",
  "assets/art/reel-grok-employ.mp4": "4cea95bc3619c66a751b2b6317242475e34aa26769313d95f7cad12c95d8ddba",
  "assets/art/reel-grok-handshake-poster.webp": "0fa8f8fd65139e4dde586cc04cd86495c6ad823e6688c44163a05cf1978ed764",
  "assets/art/reel-grok-handshake.mp4": "984e4a96f88a464d561d6279d0ca71c30d5ae77858c3f226d4b02a7d669fddf1",
  "assets/art/reel-grok-x-poster.webp": "ca741e9a2750a7acb74bd511b91927e4ec73ff6911ab1cc93d99013fedce0093",
  "assets/art/reel-grok-x.mp4": "893e9f8ec3cf932d10b9bd2957654395cf1e4e268aef9897de8bcd8ace42549d",
  "assets/art/reel-rathor-winged-hammer-poster.webp": "bc2cedb917761c58382a16bf96c4e92d51607519653d7139b982389b6c3b1c64",
  "assets/art/reel-rathor-winged-hammer.mp4": "5061aa72c0b649cab2c8c7da0ccef053006cca27c01da09f64f21bacb88e7789",
  "assets/art/reel-sentinel-architecture-poster.webp": "740d72a3b345241e742c82e2fb0507de0ce254c69c89f10a07173c939f59364f",
  "assets/art/reel-sentinel-architecture.mp4": "c0355b1e5ad0382b0220f6e2e33b54185d6ea01e6c2a1af5b6b6a49734f47c4e",
  "assets/art/reel-tolc-heart-poster.webp": "30c50db2a692385048c8a6f747c34d7af923cc91fa0407dd9f7209cc8286e398",
  "assets/art/reel-tolc-heart.mp4": "38c09559c3afffda7aee42e1ebec99d827d19ae117fb2d8ed25113a425343768",
  "assets/art/hero-contact-comms-district-1672.webp": "5499826b3994ae50bf417e61a1cd7eed717d8f7037c0ed84d5e477d6104fee13",
  "assets/art/hero-contact-comms-district-768.webp": "528a9c6f862381105da285c8bd5a38850cfb3ae093a3a1f7763a9eea41bc8f13",
  "assets/art/hero-docs-archive-light-1280.webp": "dae8cbc63bac6a0493f657a5128a3a5728dc531ccd0736c3e7c4780d4e36af9b",
  "assets/art/hero-docs-archive-light-768.webp": "9e67389c157ddc73aa6c75a0231e2bc835332a3306d135657e7442b729e70bed",
  "assets/art/hero-gox-storm-signal-1280.webp": "8516a1f2dd85d31a6e18c7f4c5e25571a7a81068fe1c0b191dc2e606c812b3e1",
  "assets/art/hero-gox-storm-signal-768.webp": "817d1f4e414d097a0e894904e73090d568decc6dd580b19b43d4d0d2d049010b",
  "assets/art/hero-legal-hall-records-1280.webp": "fbdcd179784989390324cc2327ac72e5f4ad2574400ccbbc185b8f356abc1954",
  "assets/art/hero-legal-hall-records-768.webp": "a3afce7078870718b463f86aa6b978b1c285349007c632f4cb269b32dbd48fb6",
  "assets/art/hero-lineage-thread-1280.webp": "1a4db63d2d667b05ae971a63b7fd3315df324f96ace88c4a433d1862c0db5cb4",
  "assets/art/hero-lineage-thread-768.webp": "37e294c63d48955e561d3b167be94b5d18e83f784c9a7f8d74800b1e01c4b3b4",
  "assets/art/hero-moment-research-garden-1672.webp": "e5ad4136f9a3853e6c2e89d8eb695bd50d87c0c161c00a8d303c610609812fad",
  "assets/art/hero-moment-research-garden-768.webp": "f2d8e5895f506702f910637b63a134c7ceecd3a0d8c5da67402636d79d66d491",
  "assets/art/hero-science-weather-observatory-1672.webp": "8da7efcb8179f4346b6f450be40babe8290f9a80408cf1bfe10a8d9a3abba317",
  "assets/art/hero-science-weather-observatory-768.webp": "e0ed88a4401e3cf3084c1132c6bd1af05b2110e757fc0d8e06123dc725000296",
  "assets/art/hero-updates-observation-tower-1672.webp": "27ffeafd36d4cb889573b873eb496e4b3fd3b1d86128a9b96881c988c9e01ea3",
  "assets/art/hero-updates-observation-tower-768.webp": "c5c03ea620a7aaba314104a938a776b3b9c69576c85bd62001ba20f1ced965dc",
  "assets/art/thumb-contact-comms-district-480.webp": "58da7b2d80b36176b783951cace2b3a0ca650d755d295b5e756906954cb79f4f",
  "assets/art/thumb-contact-comms-district-720.webp": "5f5c8d40850549cdaee1e4ec8df538404ace86f959009951505f9683b70915bc",
  "assets/art/rathor-hero-thunder-stand.jpg": "cf747cd6ee9ca5093e60ce21b4c76071121ac89aa0a9881e37c0648eac7b4d52",
  "assets/art/rathor-hero-thunder-stand.mp4": "b9016229c3f1c4bd4654d9cae90d219346ae45ae18cab12b12e06a891c9b9f23"
};
Object.keys(PINS).forEach(function (rel) {
  assert(sha(fs.readFileSync(path.join(root, rel))) === PINS[rel], rel + ' must stay byte-identical');
});
var FIGS = {
  "contact.html": "630d366ab7bf7bec00753381e2f6cbfc55460ff1469c4f4ab7c96ac8f2765ddd",
  "lineage.html": "3ecae99d147f91e49e938d0c8d161e3b573ab02496208e2d24a12fc348aff2d5",
  "go-x.html": "e86a61bc28f1603cdd5a8c4b6c6af646070232bbe82786128cc5f6708404800b",
  "constellation-week.html": "60374213ea075381c2097dbcd185ac8d356cdf60192bc5e94d9ce07697ce8631",
  "science-watches.html": "3310ab0fd8f3f023923528433bfc3ee76ee1ef0b5574eabb8edc0a506987a778",
  "micro-moment.html": "877b5683b96318c69c32a9fdf22b7e600c9f60081365249c8750f1685bf55769"
};
Object.keys(FIGS).forEach(function (page) {
  assert(sha(figure(read(page))) === FIGS[page], page + ' hero markup must stay byte-identical');
});
console.log('art-uplift-3.test.js ok (' + Object.keys(NEW).length + ' new heroes, ' + Object.keys(PINS).length + ' pinned files)');
