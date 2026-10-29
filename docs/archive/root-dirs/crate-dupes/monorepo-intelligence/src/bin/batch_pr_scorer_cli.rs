use monorepo_intelligence::batch_pr_scorer::{BatchPrScorer, ChangedFile, PrRecommendation};
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
        PrRecommendation::Focused => {
            println!("## Automated Batch PR Scoring\n\n**Recommendation:** Focused PR\n\nThis change appears well-suited as a focused, single-unit PR according to the Eternal Iteration Protocol.\n\n**Why Focused?**\n- Limited scope\n- Low cross-crate impact\n- Easy to review in one sitting\n\nThunder locked in.");
        }
        PrRecommendation::Batch { reason } => {
            println!("## Automated Batch PR Scoring\n\n**Recommendation:** Batch PR\n\n{}
\n**Why Batch?**\n- Multiple related files\n- Cross-crate or thematic connections\n- More efficient to review together\n\nConsider grouping these changes into one larger, coherent PR per the Batch PR Workflow guidelines in the protocol.\n\nThunder locked in.", reason);
        }
    }
}
