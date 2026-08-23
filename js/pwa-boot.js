window.__rtPwa=window.__rtPwa||{ev:null};
window.addEventListener('beforeinstallprompt',function(e){e.preventDefault();window.__rtPwa.ev=e;});
window.addEventListener('DOMContentLoaded',function(){
  if(!document.querySelector('script[src*="lang-offer"]')){
    var s=document.createElement('script'); s.src='/js/lang-offer.js'; s.defer=true; document.head.appendChild(s);
  }
});
