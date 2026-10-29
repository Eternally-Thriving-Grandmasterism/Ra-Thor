import MercyOrchestrator from '../mercy-orchestrator.js';

export default function App({ Component, pageProps }) {
  return (
    <MercyOrchestrator>
      <Component {...pageProps} />
    </MercyOrchestrator>
  );
}
