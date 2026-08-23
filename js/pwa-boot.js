window.__rtPwa=window.__rtPwa||{ev:null};
window.addEventListener('beforeinstallprompt',function(e){e.preventDefault();window.__rtPwa.ev=e;});
try{document.documentElement.setAttribute('data-theme',localStorage.getItem('rathor-theme')||'dark');}catch(e){document.documentElement.setAttribute('data-theme','dark');}
window.addEventListener('DOMContentLoaded',function(){
  if(!document.querySelector('script[src*="lang-offer"]')){
    var s=document.createElement('script'); s.src='/js/lang-offer.js'; s.defer=true; document.head.appendChild(s);
  }
  if(!document.querySelector('script[src*="rathor-theme.js"]')){
    var t=document.createElement('script'); t.src='/js/rathor-theme.js'; document.head.appendChild(t);
  }
});
