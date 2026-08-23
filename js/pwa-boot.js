window.__rtPwa=window.__rtPwa||{ev:null};
window.addEventListener('beforeinstallprompt',function(e){e.preventDefault();window.__rtPwa.ev=e;});
if('serviceWorker' in navigator){navigator.serviceWorker.register('/sw.js',{scope:'/',updateViaCache:'none'}).catch(function(){});}
window.addEventListener('DOMContentLoaded',function(){
  if(document.querySelector('script[src*="lang-offer"]')) return;
  var s=document.createElement('script');
  s.src='/js/lang-offer.js';
  s.defer=true;
  document.head.appendChild(s);
});
