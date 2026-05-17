/* ... existing full v2.5 code with all previous integrations ... */
// v2.6 update: WebRTC + DIDComm + Polygon ID zk integrated
 import { WebRTCLanMesh } from './webrtc-lan-mesh.js';
import { DIDCommV2Messaging } from './didcomm-v2-messaging.js';
import { PolygonIdZkBridge } from './polygon-id-zk-bridge.js';

// In constructor:
this.webrtc = new WebRTCLanMesh(this.mesh);
this.didcomm = new DIDCommV2Messaging(this.didBridge);
this.polygonZk = new PolygonIdZkBridge(this.didBridge);

// New methods for LAN + zk
async startLanHost() { return this.webrtc.startHost(); }
async joinLanPeer(offer) { return this.webrtc.joinPeer(offer); }
async sendSecureDidMessage(toDid, msg) { return this.didcomm.sendEncryptedMessage(toDid, msg); }
async issueZkSovereignProof(proposal) { return this.polygonZk.generateSovereignSparkZkProof(proposal); }