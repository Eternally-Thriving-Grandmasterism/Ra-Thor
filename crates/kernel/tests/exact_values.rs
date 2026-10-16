use kernel::PACKAGE_VERSION_SOURCE;

#[test]
fn package_version_is_the_manifest_string() {
    assert_eq!(PACKAGE_VERSION_SOURCE, "14.15.0");
    assert_eq!(PACKAGE_VERSION_SOURCE.len(), 7);
}
