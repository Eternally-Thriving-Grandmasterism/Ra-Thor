import { libp2p } from 'libp2p'; // Browser-native bundle
import { noise } from '@chainsafe/libp2p-noise';
import { mplex } from '@libp2p/mplex';
import { webRTCStar } from '@libp2p/webrtc-star';
import { PlonkProver } from './post-quantum-zk.js'; // Halo2-inspired
import { MercyCore } from './ra-thor-mercy-core.js';

const mercy = new MercyCore();

async function initP2PMesh(playerId) {
  const node = await libp2p.create({
    addresses: { listen: ['/webrtc-star'] },
    transports: [webRTCStar()],
    connectionEncryption: [noise()],
    streamMuxers: [mplex()],
  });

  node.handle('/powrush-valence-sync/1.0', async ({ stream }) => {
    const data = await stream.read(); // Proposed world delta
    const proof = data.proof; // Plonk valence proof
    const verified = await PlonkProver.verify(proof, data.stateHash);

    if (verified && (await mercy.valenceCheckDelta(data.delta)) >= 0.99) {
      applySyncedDelta(data.delta);
      await stream.write({ status: 'accepted' });
    } else {
      await stream.write({ status: 'rejected-mercy' });
    }
  });

  return node;
}
