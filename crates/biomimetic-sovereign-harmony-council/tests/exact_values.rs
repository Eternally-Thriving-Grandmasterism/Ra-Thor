//! Exact package version string. Not a measurement.

use biomimetic_sovereign_harmony_council::PACKAGE_VERSION_SOURCE;

#[test]
fn package_version_is_the_manifest_string() {
    assert_eq!(PACKAGE_VERSION_SOURCE, "0.1.0");
    assert_eq!(PACKAGE_VERSION_SOURCE.len(), 5);
}
