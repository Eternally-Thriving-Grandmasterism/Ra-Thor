use kernel::{PACKAGE_VERSION_SOURCE, STANDALONE_BOUNDARY};

#[test]
fn package_version_and_boundary_are_source_strings() {
    assert_eq!(PACKAGE_VERSION_SOURCE, "14.15.0");
    assert_eq!(PACKAGE_VERSION_SOURCE.len(), 7);
    assert_eq!(
        STANDALONE_BOUNDARY,
        "lattice-conductor-v14 is not a path dependency; v14 boot is not compiled"
    );
    assert_eq!(
        STANDALONE_BOUNDARY.matches("lattice-conductor-v14").count(),
        1
    );
}
