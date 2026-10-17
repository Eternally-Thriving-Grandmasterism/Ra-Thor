#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const vm = require('vm');

const ROOT = path.join(__dirname, '..');
const I18N = path.join(ROOT, 'i18n');
const LANGS =
  'ar de el es fa fr he hi id it ja ko nl pl pt ru sv th tr uk vi zh'.split(' ');

const KEYS = [
  'faqA2',
  'faqA5',
  'faqA16',
  'contactInquiryLead',
  'employFBody',
  'employOrgWho',
  'licenceWhoBody',
  'licenceFaqQuote',
];

function loadPack(filePath, lang) {
  const src = fs.readFileSync(filePath, 'utf8');
  const ctx = { window: { translations: {} } };
  vm.createContext(ctx);
  vm.runInContext(src, ctx);
  return ctx.window.translations[lang];
}

function escJs(value) {
  return JSON.stringify(value).slice(1, -1);
}

function setKey(src, key, value) {
  const esc = escJs(value);
  const re = new RegExp(
    `("${key.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}"\\s*:\\s*)"(?:[^"\\\\]|\\\\.)*"`,
  );
  if (!re.test(src)) {
    throw new Error(`Key not found: ${key}`);
  }
  return src.replace(re, `$1"${esc}"`);
}

const en = loadPack(path.join(I18N, 'en.js'), 'en');
const licenceFaqQuote = en.licenceFaqQuote;
const translations = require('./rw1-translations.cjs');

for (const lang of LANGS) {
  const filePath = path.join(I18N, `${lang}.js`);
  let src = fs.readFileSync(filePath, 'utf8');
  const pack = translations[lang];
  if (!pack) {
    throw new Error(`Missing translations for ${lang}`);
  }
  for (const key of KEYS) {
    const value =
      key === 'licenceFaqQuote' ? licenceFaqQuote : pack[key];
    if (value == null || value === '') {
      throw new Error(`Missing ${key} for ${lang}`);
    }
    src = setKey(src, key, value);
  }
  fs.writeFileSync(filePath, src, 'utf8');
  console.log(`updated i18n/${lang}.js`);
}

console.log('done');
