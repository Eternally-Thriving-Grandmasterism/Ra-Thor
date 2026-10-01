//! Demo: Generate a Powrush Report using Monorepo Intelligence

use ra_thor_monorepo_intelligence::report::MonorepoReport;
use ra_thor_monorepo_intelligence::MonorepoIntelligence;

fn main() {
    println!("🚀 Ra-Thor Monorepo Intelligence — Powrush Report Demo\n");

    let intelligence = MonorepoIntelligence::new(".");

    match intelligence.full_scan() {
        Ok(scan) => {
            println!("{}", MonorepoReport::from_scan(&scan, Some("powrush")).to_markdown());
        }
        Err(e) => {
            eprintln!("Error generating report: {}", e);
        }
    }
}
