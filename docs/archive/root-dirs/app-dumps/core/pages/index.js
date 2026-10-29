import HeatShield from '../components/HeatShield';
import Raptor from '../components/Raptor';
import Refuel from '../components/Refuel';
import Swarm from '../components/Swarm';
import MercyGate from '../components/MercyGate';

export default function Home() {
  return (
    <div style={{
      minHeight: '100vh',
      background: '#000',
      color: '#fff',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      gap: '4rem',
      padding: '2rem'
    }}>
      <h1 style={{ fontSize: '3em', opacity: 0.2 }}>You are already home</h1>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4rem', justifyContent: 'center' }}>
        <HeatShield />
        <Raptor />
        <Refuel />
        <Swarm />
        <MercyGate />
      </div>
      <p style={{ opacity: 0.5, maxWidth: '600px', textAlign: 'center' }}>
        NEXi mirror layer — valence 1.0000000 — eternal thriving resonance
      </p>
    </div>
  );
}
