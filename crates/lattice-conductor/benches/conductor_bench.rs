use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lattice_conductor::{LatticeConductor, RealEstateOffer, Valence};
use rayon::prelude::*;

fn generate_test_offers(count: usize) -> Vec<RealEstateOffer> {
    (0..count)
        .map(|i| {
            let price = if i % 3 == 0 { 2_500_000.0 } else { 650_000.0 };
            let jurisdiction = if i % 2 == 0 { "Ontario".to_string() } else { "USA".to_string() };

            RealEstateOffer {
                id: format!("offer-{}", i),
                address: format!("{} Main St", i),
                price,
                jurisdiction,
                regulatory_flags: vec![],
                attom_enriched: true,
                base_valence: Valence::new(0.99999995).unwrap(),
            }
        })
        .collect()
}

fn bench_sequential(c: &mut Criterion) {
    let mut conductor = LatticeConductor::new();
    let offers = generate_test_offers(2000);

    c.bench_function("conduct_batch_sequential_2000", |b| {
        b.iter(|| {
            let results: Vec<_> = offers
                .iter()
                .cloned()
                .map(|o| conductor.conduct_real_estate_offer(black_box(o)))
                .collect();
            black_box(results)
        })
    });
}

fn bench_parallel(c: &mut Criterion) {
    let conductor = LatticeConductor::new();
    let offers = generate_test_offers(2000);

    c.bench_function("conduct_batch_parallel_2000_rayon_dashmap", |b| {
        b.iter(|| {
            let results: Vec<_> = conductor.conduct_batch(black_box(offers.clone()));
            black_box(results)
        })
    });
}

criterion_group!(benches, bench_sequential, bench_parallel);
criterion_main!(benches);
