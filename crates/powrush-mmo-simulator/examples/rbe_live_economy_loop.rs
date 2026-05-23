// Live RBE Economy Loop with Mercy Gating
// Phase 3 integration

use mercy_gating_runtime::{BeingRace, MercyGate16Numeric, MaAtResonance, pipeline_passes_numeric_with_ma_at, process_rbe_transaction};

fn run_live_rbe_economy_loop() {
    println!("=== POWRUSH-MMO LIVE RBE ECONOMY LOOP (MERCY-GATED) ===\n");

    let mut economy = RBEconomy::new();
    let mut turn = 1;

    loop {
        println!("--- Turn {} ---", turn);

        let tx = ResourceTransaction { amount: 120.0, from: "Druid Enclave", to: "Starborn Collective" };
        
        let mercy_ok = pipeline_passes_numeric_with_ma_at(&current_gates, &current_ma_at, Some(BeingRace::Druid));
        
        let result = if mercy_ok {
            process_rbe_transaction(&tx, &current_gates, &current_ma_at, Some(BeingRace::Druid))
        } else {
            RBETransactionResult::BlockedByMercy { reason: "Gate violation — flow redirected to healing" }
        };

        economy.apply_result(result);
        println!("RBE Flow processed. Mercy status: {}", if mercy_ok { "PASSED" } else { "HEALING CYCLE" });

        if turn >= 8 { break; }
        turn += 1;
    }
}