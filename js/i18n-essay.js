/* js/i18n-essay.js
 * Visitor essay apply. Loads after js/i18n-chrome.js.
 * Workspace 14.15.6 · info@Rathor.ai
 * FAQ, Employ, Privacy, briefing, contact guidance, and
 * launch / go-x / science short prose. Operator docs stay English.
 * Missing key → English. Never blank. Never invent METR.
 * A real translation (string differs from English) sets that
 * article dir/lang to the applied language. An English copy
 * stays dir=ltr lang=en. Family pills and language tabs stay LTR.
 * Does not set html[dir]. Does not touch COEP or Google Translate.
 */
(function (root) {
  'use strict';
  if (root.__rtI18nEssay) return;
  root.__rtI18nEssay = true;

  var ESSAY = {
    /* ESSAY_KEYS_START */
    briefBack: 1,
    briefClosePaid: 1,
    briefDoor1: 1,
    briefDoor2: 1,
    briefDoor3: 1,
    briefDoor4: 1,
    briefDoorsLead: 1,
    briefDoorsTitle: 1,
    briefLead1: 1,
    briefLead2: 1,
    briefLoop2: 1,
    briefLoop3: 1,
    briefLoop4: 1,
    briefLoopNote: 1,
    briefLoopTitle: 1,
    briefMonths1: 1,
    briefMonths2: 1,
    briefMonths3: 1,
    briefMonthsTitle: 1,
    briefOrgHabits: 1,
    briefOrgPeace: 1,
    briefPrompt: 1,
    briefStart1: 1,
    briefStart2: 1,
    briefStart3: 1,
    briefStart4: 1,
    briefStart5: 1,
    briefStartFoot: 1,
    briefStartTitle: 1,
    briefSubtitle: 1,
    briefThird1: 1,
    briefThird2: 1,
    briefThirdTitle: 1,
    briefTitle: 1,
    briefUseCursor: 1,
    briefUseEngineer: 1,
    briefUseGame: 1,
    briefUseOrg: 1,
    briefUseTeacher: 1,
    briefUseTitle: 1,
    briefUseWriter: 1,
    briefWhat1: 1,
    briefWhat2: 1,
    briefWhatTitle: 1,
    contactGuide1: 1,
    contactGuide2: 1,
    contactGuide3: 1,
    contactGuide4: 1,
    contactGuide5: 1,
    contactGuideNote: 1,
    contactGuideSubject: 1,
    contactGuideTitle: 1,
    contactInquiryFoot: 1,
    contactInquiryLead: 1,
    contactInquiryTitle: 1,
    contactMessageLabel: 1,
    contactNameLabel: 1,
    contactOrgLabel: 1,
    contactReplyLabel: 1,
    contactSeatsLabel: 1,
    contactSubmit: 1,
    contactSubmitStatus: 1,
    contactUrlLabel: 1,
    contactUseEval: 1,
    contactUseLegend: 1,
    contactUseOther: 1,
    contactUseShip: 1,
    contactUseWrap: 1,
    demoNotProduct: 1,
    employA1: 1,
    employA2: 1,
    employA3: 1,
    employA4: 1,
    employA5: 1,
    employA6: 1,
    employATitle: 1,
    employB1: 1,
    employB10: 1,
    employB11: 1,
    employB12: 1,
    employB2: 1,
    employB3: 1,
    employB4: 1,
    employB5: 1,
    employB6: 1,
    employB7: 1,
    employB8: 1,
    employB9: 1,
    employBFoot: 1,
    employBTitle: 1,
    employCInspect: 1,
    employCInspectBody: 1,
    employCLead: 1,
    employCOrg: 1,
    employCOrgBody: 1,
    employCPersonal: 1,
    employCPersonalBody: 1,
    employCTitle: 1,
    employClose1: 1,
    employClose2: 1,
    employDNote: 1,
    employDTitle: 1,
    employDoorBrief: 1,
    employDoorChat: 1,
    employDoorPilot: 1,
    employETitle: 1,
    employEWalk: 1,
    employFBody: 1,
    employFTitle: 1,
    employGBody: 1,
    employGTitle: 1,
    employLoop1: 1,
    employLoop2: 1,
    employLoop3: 1,
    employLoop4: 1,
    employLoop5: 1,
    employLoopFoot: 1,
    employLoopTitle: 1,
    employOrgBuy: 1,
    employOrgHow: 1,
    employOrgNot: 1,
    employOrgTitle: 1,
    employOrgWho: 1,
    employSisterBody: 1,
    employSisterTitle: 1,
    employSpine: 1,
    employStamp: 1,
    employWrap1: 1,
    employWrap2: 1,
    employWrap3: 1,
    employWrap4: 1,
    employWrapFoot: 1,
    employWrapLead: 1,
    employWrapTitle: 1,
    footerFamilyTitle: 1,
    footerPrivacyTitle: 1,
    footerWorkspaceTitle: 1,
    goxNote: 1,
    goxPrimary: 1,
    goxRelay: 1,
    goxWindow: 1,
    homeCopyright: 1,
    homeFooterPrivacy: 1,
    homeFooterWorkspace: 1,
    homeLaunchMap: 1,
    homeMoments: 1,
    homeMonorepo: 1,
    homeOfflineChat: 1,
    homePaperLink: 1,
    homePaperNote: 1,
    homePwaNote: 1,
    homeRecentLink: 1,
    homeRepoNote: 1,
    homeStatusPowrush: 1,
    homeStatusTolc: 1,
    homeStatusVersion: 1,
    launchAirNote: 1,
    launchAirTitle: 1,
    launchAnchorBody: 1,
    launchAnchorKicker: 1,
    launchAnchorTitle: 1,
    launchArkNote: 1,
    launchArkTitle: 1,
    launchChatBody: 1,
    launchChatCta: 1,
    launchChatKicker: 1,
    launchChatMeta: 1,
    launchChatTitle: 1,
    launchCouncil: 1,
    launchEmployBody: 1,
    launchEmployCta: 1,
    launchEmployKicker: 1,
    launchEmployMeta: 1,
    launchEmployTitle: 1,
    launchForgeBody: 1,
    launchForgeCta: 1,
    launchForgeKicker: 1,
    launchForgeMeta: 1,
    launchForgeTitle: 1,
    launchFusionNote: 1,
    launchFusionTitle: 1,
    launchHtcNote: 1,
    launchHtcTitle: 1,
    launchKickerAgsi: 1,
    launchKickerMap: 1,
    launchKickerWs: 1,
    launchLead: 1,
    launchMercyNote: 1,
    launchMercyTitle: 1,
    launchOneBody: 1,
    launchOneCta: 1,
    launchOneKicker: 1,
    launchOneMeta: 1,
    launchOneTitle: 1,
    launchProteinNote: 1,
    launchProteinTitle: 1,
    launchRunLead: 1,
    launchRunNote: 1,
    launchRunTitle: 1,
    launchScienceLead: 1,
    launchScienceTitle: 1,
    launchShardBody: 1,
    launchShardCta: 1,
    launchShardKicker: 1,
    launchShardMeta: 1,
    launchShardTitle: 1,
    launchStudyGpuNote: 1,
    launchStudyGpuTitle: 1,
    launchStudyLead: 1,
    launchStudyMeshNote: 1,
    launchStudyMeshTitle: 1,
    launchStudyMomentNote: 1,
    launchStudyMomentTitle: 1,
    launchStudyTitle: 1,
    launchStudyWatchNote: 1,
    launchStudyWeekNote: 1,
    launchStudyWeekTitle: 1,
    launchStudyXNote: 1,
    launchStudyXTitle: 1,
    launchSubtitle: 1,
    launchTitle: 1,
    paperKicker: 1,
    privacyClosing: 1,
    privacyComp1: 1,
    privacyComp2: 1,
    privacyComp3: 1,
    privacyComp4: 1,
    privacyComp5: 1,
    privacyComplianceFoot: 1,
    privacyComplianceLead: 1,
    privacyComplianceTitle: 1,
    privacyContactIssues: 1,
    privacyContactLead: 1,
    privacyContactMail: 1,
    privacyContactTitle: 1,
    privacyData1: 1,
    privacyData2: 1,
    privacyData3: 1,
    privacyData4: 1,
    privacyData5: 1,
    privacyData6: 1,
    privacyDataFoot: 1,
    privacyDataLead: 1,
    privacyDataTitle: 1,
    privacyHow1: 1,
    privacyHow2: 1,
    privacyHowTitle: 1,
    privacyLead: 1,
    privacyOverview1: 1,
    privacyOverview2: 1,
    privacyOverviewTitle: 1,
    privacyRight1: 1,
    privacyRight2: 1,
    privacyRight3: 1,
    privacyRight4: 1,
    privacyRight5: 1,
    privacyRightsLead: 1,
    privacyRightsTitle: 1,
    privacyThird1: 1,
    privacyThird2: 1,
    privacyThirdTitle: 1,
    privacyUpdated: 1,
    swatchCalBody: 1,
    swatchCalLink: 1,
    swatchCalTitle: 1,
    swatchFoot: 1,
    swatchIsecBody: 1,
    swatchIsecLink: 1,
    swatchIsecTitle: 1,
    swatchLead: 1,
    swatchStamp: 1,
    swatchTitle: 1,
    /* ESSAY_KEYS_END */
  };

  function isEssayKey(key) {
    if (!key) return false;
    if (ESSAY[key]) return true;
    return /^faq(Title|Contact|Q\d+|A\d+)$/.test(key);
  }

  function packOf(lang) {
    var packs = root.translations || {};
    return packs[lang] || {};
  }

  function pick(lang, key) {
    if (typeof root.rtPickChrome === 'function') return root.rtPickChrome(lang, key);
    var pack = packOf(lang);
    var en = packOf('en');
    var val = pack[key];
    if (val != null && String(val) !== '') return { val: val, fallback: false };
    if (en[key] != null && String(en[key]) !== '') return { val: en[key], fallback: lang !== 'en' };
    return { val: null, fallback: true };
  }

  function isRtlText(s) {
    if (typeof root.rtIsRtlText === 'function') return root.rtIsRtlText(s);
    return typeof s === 'string' && /[\u0590-\u08FF\uFB1D-\uFDFF\uFE70-\uFEFF]/.test(s);
  }

  function isReal(lang, key, picked) {
    if (!picked || picked.val == null || String(picked.val) === '') return false;
    if (lang === 'en') return true;
    var enVal = packOf('en')[key];
    if (enVal != null && String(picked.val) === String(enVal)) return false;
    if (picked.fallback) return false;
    return true;
  }

  function setEssayDir(el, lang, real, rtl) {
    if (!el || !el.setAttribute) return;
    if (!real) {
      el.setAttribute('dir', 'ltr');
      el.setAttribute('lang', 'en');
      return;
    }
    el.setAttribute('dir', rtl ? 'rtl' : 'ltr');
    el.setAttribute('lang', lang || 'en');
  }

  function containerOf(el) {
    if (!el || !el.closest) return null;
    return el.closest('article, #faq, [data-rt-prose]');
  }

  function keepFamilyLtr() {
    var family = document.getElementById('rt-family-nav');
    if (family) family.setAttribute('dir', 'ltr');
    var langBar = document.getElementById('lang-selector');
    if (langBar) langBar.setAttribute('dir', 'ltr');
  }

  function applyEssayI18n(lang) {
    lang = lang || 'en';
    if (!document.querySelectorAll) return;
    var buckets = [];
    function bucketFor(container) {
      for (var i = 0; i < buckets.length; i++) {
        if (buckets[i].el === container) return buckets[i];
      }
      var created = { el: container, real: 0, fallback: 0, rtl: 0 };
      buckets.push(created);
      return created;
    }

    var nodes = document.querySelectorAll('[data-i18n], [data-lock-i18n]');
    for (var n = 0; n < nodes.length; n++) {
      var el = nodes[n];
      var key = el.getAttribute('data-i18n') || el.getAttribute('data-lock-i18n');
      if (!isEssayKey(key)) continue;
      if (typeof root.rtIsChromeKey === 'function' && root.rtIsChromeKey(key)) continue;
      var picked = pick(lang, key);
      if (picked.val == null || String(picked.val) === '') continue;
      var real = isReal(lang, key, picked);
      var asHtml = el.hasAttribute('data-i18n-html') || String(picked.val).indexOf('<') !== -1;
      if (asHtml) el.innerHTML = picked.val;
      else el.textContent = picked.val;
      var keepLtr = el.hasAttribute('data-rt-keep-ltr');
      var rtl = real && !keepLtr && isRtlText(picked.val);
      if (keepLtr) {
        el.setAttribute('dir', 'ltr');
        el.setAttribute('lang', real ? (lang || 'en') : 'en');
      } else {
        setEssayDir(el, lang, real, rtl);
      }
      var box = containerOf(el);
      if (!box || box === el) continue;
      var bucket = bucketFor(box);
      if (real) bucket.real++;
      else bucket.fallback++;
      if (rtl) bucket.rtl++;
    }

    for (var b = 0; b < buckets.length; b++) {
      var item = buckets[b];
      var useReal = item.real > 0 && item.real >= item.fallback;
      var boxRtl = useReal && item.rtl > 0 && item.rtl * 2 >= item.real;
      setEssayDir(item.el, lang, useReal, boxRtl);
    }

    keepFamilyLtr();
  }

  root.rtIsEssayKey = isEssayKey;
  root.rtApplyEssayI18n = applyEssayI18n;

  document.addEventListener('rt-chrome-i18n', function (e) {
    var lang = e && e.detail && e.detail.lang;
    applyEssayI18n(lang || 'en');
  });

  function bootEssay() {
    var lang = 'en';
    try { lang = localStorage.getItem('rathor-lang') || 'en'; } catch (err) { lang = 'en'; }
    if (root.translations && root.translations[lang]) applyEssayI18n(lang);
    else if (root.translations && root.translations.en) applyEssayI18n('en');
    keepFamilyLtr();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', bootEssay);
  else bootEssay();
})(typeof window !== 'undefined' ? window : this);
