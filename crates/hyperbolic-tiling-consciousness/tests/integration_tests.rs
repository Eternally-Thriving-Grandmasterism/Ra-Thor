use hyperbolic_tiling_consciousness::{
    generate_regular_tiling, hyperbolic_distance, PoincarePoint,
};

#[test]
fn distance_matches_the_poincare_formula() {
    let a = PoincarePoint::new(0.1, 0.0);
    let b = PoincarePoint::new(0.2, 0.0);
    let num = 2.0 * ((a.x - b.x).powi(2) + (a.y - b.y).powi(2));
    let den = (1.0 - a.r2()) * (1.0 - b.r2());
    let expect = (1.0 + num / den).acosh();
    assert!((hyperbolic_distance(&a, &b) - expect).abs() < 1e-12);
}

#[test]
fn five_layers_keep_the_points_the_loop_builds() {
    assert_eq!(generate_regular_tiling(5).len(), 217);
}
