/* license-inquiry.js — LICENSE-FORM-1
 * Submit builds mailto:info@Rathor.ai?subject=AG-SML%20commercial%20inquiry
 * Opens the visitor’s mail app. Does not upload to rathor.ai.
 * No cookies, no backend, no checkout, no payment processor.
 */
(function (global) {
  'use strict';

  var MAILTO = 'info@Rathor.ai';
  var SUBJECT = 'AG-SML commercial inquiry';

  function text(value) {
    return String(value == null ? '' : value).trim();
  }

  function line(label, value) {
    return label + ': ' + text(value);
  }

  function encodeMailto(fields) {
    fields = fields || {};
    var body = [
      line('Organization name', fields.organization),
      line('Public site / product URL', fields.url),
      line('Use', fields.use),
      line('Approx seats or servers', fields.seats),
      '',
      'Message:',
      text(fields.message)
    ].join('\n');
    return 'mailto:' + MAILTO +
      '?subject=' + encodeURIComponent(SUBJECT) +
      '&body=' + encodeURIComponent(body);
  }

  function readForm(form) {
    if (!form) return {};
    var useEl = form.querySelector('[name="use"]:checked') || form.querySelector('[name="use"]');
    return {
      organization: form.organization ? form.organization.value : '',
      url: form.url ? form.url.value : '',
      use: useEl ? useEl.value : '',
      seats: form.seats ? form.seats.value : '',
      message: form.message ? form.message.value : ''
    };
  }

  function bind() {
    var form = document.getElementById('license-inquiry-form');
    if (!form || form.getAttribute('data-rt-bound') === '1') return;
    form.setAttribute('data-rt-bound', '1');
    form.addEventListener('submit', function (e) {
      e.preventDefault();
      if (typeof form.reportValidity === 'function' && !form.reportValidity()) return;
      var href = encodeMailto(readForm(form));
      form.setAttribute('data-mailto', href);
      global.location.href = href;
    });
  }

  var api = {
    encodeMailto: encodeMailto,
    readForm: readForm,
    SUBJECT: SUBJECT,
    MAILTO: MAILTO
  };

  if (typeof document !== 'undefined') {
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', bind);
    else bind();
  }

  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  global.rtLicenseInquiry = api;
})(typeof window !== 'undefined' ? window : this);
