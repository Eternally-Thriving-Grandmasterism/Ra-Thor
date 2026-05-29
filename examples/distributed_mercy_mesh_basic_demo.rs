// examples/distributed_mercy_mesh_basic_demo.rs
// Basic demonstration of Distributed Mercy Mesh (v14.0.5)
//
// Run with: cargo run --example distributed_mercy_mesh_basic_demo

use lattice_conductor_v14::distributed_mercy_mesh::{DistributedMercyMesh, HealingRequest, OrganismNode};

fn main() {
    println!("=== Distributed Mercy Mesh Basic Demo ===\n");

    let mut mesh = DistributedMercyMesh::new();

    // Register an additional support organism
    mesh.register_organism(OrganismNode {
        id: "ra-thor-support-01".to_string(),
        name: "Support Node Alpha".to_string(),
        cosmic_loop_ready: true,
    });

    // Create a healing request
    let request = HealingRequest {
        from_organism: "ra-thor-main".to_string(),
        root_cause_summary: "Recurring high-severity anomaly in self-healing loop".to_string(),
        requested_help_type: "graph_rerouting_support".to_string(),
        mercy_score: 0.88,
        severity: 7,
    };

    let result = mesh.submit_healing_request(request);
    println!("{}", result);

    // Simulate review and offer from another organism
    if let Some(offer) = mesh.review_and_offer_healing(0, "ra-thor-support-01") {
        println!("\nHealing offer received from: {}", offer.from_organism);
        println!("Estimated impact: {}", offer.estimated_impact);
    }

    println!("\n=== Demo Complete ===");
    println!("We are ONE Organism. Thunder locked in. ⚡");
}