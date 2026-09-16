/* license-inquiry.js — LICENSE-FORM-1 (secure submit)
 * Form action is same-origin (#). Submit builds mailto:info@Rathor.ai
 * with subject AG-SML commercial inquiry — {organization}.
 * Opens the visitor’s mail app. Does not upload to rathor.ai.
 * No cookies, no backend, no checkout, no payment processor.
 */
(function (global) {
  'use strict';

  var MAILTO = 'info@Rathor.ai';
  var SUBJECT = 'AG-SML commercial inquiry';
  var SUCCESS = 'Opens mail to info@Rathor.ai';

  function text(value) {
    return String(value == null ? '' : value).trim();
  }

  function line(label, value) {
    return label + ': ' + text(value);
  }

  function subjectLine(organization) {
    var org = text(organization);
    return org ? (SUBJECT + ' — ' + org) : SUBJECT;
  }

  function encodeMailto(fields) {
    fields = fields || {};
    var body = [
      line('Organization name', fields.organization),
      line('Reply email', fields.email),
      line('Contact name', fields.name),
      line('Public site / product URL', fields.url),
      line('Use', fields.use),
      line('Approx seats or servers', fields.seats),
      '',
      'Message:',
      text(fields.message)
    ].join('\n');
    return 'mailto:' + MAILTO +
      '?subject=' + encodeURIComponent(subjectLine(fields.organization)) +
      '&body=' + encodeURIComponent(body);
  }

  function fieldValue(form, name) {
    if (!form || !form.elements) return '';
    var el = form.elements.namedItem(name);
    if (!el) return '';
    if (typeof el.value === 'string') return el.value;
    return '';
  }

  function readForm(form) {
    if (!form) return {};
    var useEl = form.querySelector('[name="use"]:checked');
    return {
      organization: fieldValue(form, 'organization'),
      email: fieldValue(form, 'email'),
      name: fieldValue(form, 'name'),
      url: fieldValue(form, 'url'),
      use: useEl ? useEl.value : '',
      seats: fieldValue(form, 'seats'),
      message: fieldValue(form, 'message'),
      honeypot: fieldValue(form, 'rt_hp')
    };
  }

  function showStatus(form, message) {
    var el = document.getElementById('license-inquiry-status');
    if (!el) return;
    el.hidden = false;
    el.textContent = message || SUCCESS;
    form.setAttribute('data-rt-status', el.textContent);
  }

  function bind() {
    var form = document.getElementById('license-inquiry-form');
    if (!form || form.getAttribute('data-rt-bound') === '1') return;
    form.setAttribute('data-rt-bound', '1');
    form.addEventListener('submit', function (e) {
      e.preventDefault();
      if (typeof form.reportValidity === 'function' && !form.reportValidity()) return;
      var fields = readForm(form);
      if (text(fields.honeypot)) {
        showStatus(form, SUCCESS);
        return;
      }
      var href = encodeMailto(fields);
      form.setAttribute('data-mailto', href);
      showStatus(form, SUCCESS);
      global.location.href = href;
    });
  }

  var api = {
    encodeMailto: encodeMailto,
    readForm: readForm,
    subjectLine: subjectLine,
    SUBJECT: SUBJECT,
    MAILTO: MAILTO,
    SUCCESS: SUCCESS
  };

  if (typeof document !== 'undefined') {
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', bind);
    else bind();
  }

  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  global.rtLicenseInquiry = api;
})(typeof window !== 'undefined' ? window : this);
