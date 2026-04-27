//! Demo: Generate a Powrush Report using Monorepo Intelligence

use ra_thor_monorepo_intelligence::MonorepoIntelligence;

fn main() {
    println!("🚀 Ra-Thor Monorepo Intelligence — Powrush Report Demo\n");

    let intelligence = MonorepoIntelligence::new(".");

    match intelligence.generate_powrush_report() {
        Ok(report) => {
            println!("{}", report);
        }
        Err(e) => {
            eprintln!("Error generating report: {}", e);
        }
    }
}
