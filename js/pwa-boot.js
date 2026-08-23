window.__rtPwa=window.__rtPwa||{ev:null};
window.addEventListener('beforeinstallprompt',function(e){e.preventDefault();window.__rtPwa.ev=e;});
if('serviceWorker' in navigator){navigator.serviceWorker.register('/sw.js',{scope:'/',updateViaCache:'none'}).catch(function(){});}
