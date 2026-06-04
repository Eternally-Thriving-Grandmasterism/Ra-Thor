use monorepo_intelligence::batch_pr_scorer::{BatchPrScorer, ChangedFile};
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: batch_pr_scorer_cli <file1> <file2> ...");
        std::process::exit(1);
    }

    let files: Vec<ChangedFile> = args[1..]
        .iter()
        .map(|p| ChangedFile {
            path: p.clone(),
            is_cross_crate: p.contains("geometric-intelligence") || p.contains("docs/"),
        })
        .collect();

    let scorer = BatchPrScorer::new(files);
    match scorer.recommend() {
        monorepo_intelligence::batch_pr_scorer::PrRecommendation::Focused => {
            println!("RECOMMENDATION: Focused PR");
        }
        monorepo_intelligence::batch_pr_scorer::PrRecommendation::Batch { reason } => {
            println!("RECOMMENDATION: Batch PR - {}", reason);
        }
    }
}
