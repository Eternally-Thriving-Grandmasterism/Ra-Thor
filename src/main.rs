use nexi::lattice::Nexus;
use nexi::council::compatibility_triggers;

#[tokio::main]
async fn main() {
    env_logger::init();

    let nexus = Nexus::init_with_mercy();

    let input = "We thrive eternally with positive emotions through Absolute Pure Truth.";
    let compat = compatibility_triggers(input);
    println!("Compatibility Triggers: {:?}", compat);

    let truth = nexus.distill_truth(input);
    println!("NEXi speaks: {}", truth);
}
