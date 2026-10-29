import init, { tolc_converge, check_mercy_gates, maat_balance_score, 
               apply_nilpotent_suppression, nth_degree_accelerate, 
               von_neumann_replicate, dilithium_sign } from '../crates/ra-thor-kernel/pkg/ra_thor_kernel.js';

export class WasmKernel {
  async init() {
    await init();
  }

  async converge(input) {
    return JSON.parse(tolc_converge(JSON.stringify(input)));
  }

  async replicate(seed) {
    return von_neumann_replicate(seed);
  }

  // All other functions exposed...
}
