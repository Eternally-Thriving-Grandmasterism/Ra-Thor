/* W4c: dir follows applied chrome strings; English prose stays LTR. */
var fs = require('fs');
var path = require('path');
var vm = require('vm');
var root = path.join(__dirname, '..');

function assert(cond, msg) {
  if (!cond) throw new Error(msg);
}

var attrs = {};
function el(init) {
  init = init || {};
  var node = {
    _text: init.text || '',
    _html: init.html || '',
    attrs: Object.assign({ 'data-i18n': init.key || null }, init.attrs || {}),
    classList: { remove: function () {}, toggle: function () {} },
    hasAttribute: function (k) { return node.attrs[k] != null; },
    getAttribute: function (k) { return node.attrs[k]; },
    setAttribute: function (k, v) { node.attrs[k] = v; }
  };
  Object.defineProperty(node, 'textContent', {
    get: function () { return node._text; },
    set: function (v) { node._text = v; }
  });
  Object.defineProperty(node, 'innerHTML', {
    get: function () { return node._html || node._text; },
    set: function (v) { node._html = v; node._text = String(v).replace(/<[^>]+>/g, ''); }
  });
  return node;
}

var employTitle = el({ key: 'employTitle', text: 'How to employ' });
var weekMore = el({ key: 'weekMore', text: 'Full week note →' });
var faqA1 = el({ key: 'faqA1', text: 'Yes, for personal use under AG-SML v1.1.' });
var prose = el({ attrs: { class: 'rt-prose' }, text: 'Ra-Thor is inspectable research software from Autonomicity Games Inc.' });
prose.classList = { remove: function () {}, toggle: function () {} };
var html = el({ attrs: { lang: 'en', dir: 'ltr' } });
var main = el({ attrs: { id: 'rt-family-main' } });
var nodes = [employTitle, weekMore, faqA1];

var document = {
  documentElement: html,
  querySelectorAll: function (sel) {
    if (sel.indexOf('data-i18n') !== -1) return nodes;
    if (sel.indexOf('rt-prose') !== -1 || sel.indexOf('article') !== -1) return [prose];
    if (sel.indexOf('lang-tab') !== -1) return [];
    return [];
  },
  querySelector: function (sel) {
    if (sel.indexOf('kicker') !== -1) return null;
    if (sel.indexOf('main') !== -1 || sel.indexOf('rt-family-main') !== -1) return main;
    return null;
  },
  getElementById: function (id) { return id === 'faq' ? null : null; },
  dispatchEvent: function () {}
};

var ctx = {
  window: {},
  document: document,
  translations: {
    en: {
      employTitle: 'How to employ',
      employSubtitle: 'Three doors.',
      weekMore: 'Full week note →',
      faqA1: 'Yes, English FAQ.'
    },
    ar: {
      weekMore: 'ملاحظة الأسبوع كاملة ←',
      faqA1: 'نعم، إجابة عربية طويلة لا يجب أن تُطبَّق.'
    }
  },
  localStorage: { setItem: function () {}, getItem: function () { return 'ar'; } },
  CustomEvent: function (name, opts) { this.name = name; this.detail = opts && opts.detail; }
};
ctx.window = ctx;
ctx.root = ctx;

var code = fs.readFileSync(path.join(root, 'js/i18n-chrome.js'), 'utf8');
vm.runInNewContext(code + '\nthis.rtApplyChromeI18n("ar");', ctx);

assert(ctx.rtIsRtlText('ملاحظة'), 'Arabic is RTL');
assert(!ctx.rtIsRtlText('Ra-Thor is inspectable'), 'English is not RTL');
assert(ctx.rtIsChromeKey('employTitle'), 'employTitle is chrome');
assert(ctx.rtIsLongCopyKey('faqA1'), 'faqA1 is long copy');
assert(ctx.rtIsLongCopyKey('weekLineResearch'), 'week footnote is long copy');
assert(faqA1.textContent.indexOf('Yes, for personal') === 0, 'FAQ answer stays English HTML');
assert(faqA1.attrs.dir === 'ltr', 'FAQ answer dir=ltr');
assert(employTitle.textContent === 'How to employ', 'missing ar employTitle falls back to English');
assert(employTitle.attrs.dir === 'ltr', 'English fallback forces ltr on the title');
assert(weekMore.attrs.dir === 'rtl', 'real Arabic chrome may be rtl');
assert(html.attrs.dir === 'ltr', 'html stays ltr when most chrome fell back');
assert(main.attrs.dir === 'ltr', 'main stays ltr when most nodes fell back');
assert(prose.attrs.dir === 'ltr', 'rt-prose stays ltr');

var gjs = fs.readFileSync(path.join(root, 'js/google-translate-optin.js'), 'utf8');
assert(gjs.indexOf('translate_a/element.js') === -1, 'W4b still has no widget inject');

var css = fs.readFileSync(path.join(root, 'css/rathor-theme-rest-a.css'), 'utf8');
assert(css.indexOf('.rt-card-uniform { display: flex; flex-direction: column; min-height: 0; }') !== -1, 'min-height 100% removed globally');

var shell = fs.readFileSync(path.join(root, 'css/rathor-home-shell.css'), 'utf8');
assert(shell.indexOf('#rt-family-nav') !== -1 && shell.indexOf('direction: ltr') !== -1, 'family nav not mirrored');

var employ = fs.readFileSync(path.join(root, 'employ.html'), 'utf8');
assert(employ.indexOf('class="rt-prose"') !== -1, 'Employ long copy wrapped');
assert(employ.indexOf('data-i18n="employTitle"') !== -1, 'Employ title is chrome');
assert(!/class="[^"]*rt-btn[^"]*"[^>]*>\s*<i class="fa-/.test(employ.replace(/\n/g, ' ')), 'Employ CTAs have no FA icons');

console.log('W4c dir-fallback checks passed');
