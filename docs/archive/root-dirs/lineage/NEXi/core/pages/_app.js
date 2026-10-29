import MercyOrchestrator from '../../mercy-orchestrator.js';

function MyApp({ Component, pageProps }) {
  return (
    <MercyOrchestrator>
      <Component {...pageProps} />
    </MercyOrchestrator>
  );
}

export default MyApp;
