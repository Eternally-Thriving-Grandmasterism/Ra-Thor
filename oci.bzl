# oci.bzl – OCI image definitions for Rathor-NEXi

load("@rules_oci//oci:defs.bzl", "oci_image", "oci_tarball")
load("@rules_pkg//:pkg.bzl", "pkg_tar")

# Final runtime image (dist/ from Vite build + nginx)
pkg_tar(
    name = "vite_dist_layer",
    srcs = [":build"],
    package_dir = "/usr/share/nginx/html",
)

oci_image(
    name = "rathor_image",
    base = "@nginx_alpine",  # or distroless/static
    tars = [":vite_dist_layer"],
    entrypoint = ["/usr/sbin/nginx"],
    cmd = ["-g", "daemon off;"],
    exposed_ports = ["80/tcp"],
    labels = {
        "org.opencontainers.image.source": "https://github.com/Eternally-Thriving-Grandmasterism/Rathor-NEXi",
        "org.opencontainers.image.description": "Rathor NEXi – Sovereign offline AGI lattice",
        "org.opencontainers.image.licenses": "MIT",
    },
)

oci_tarball(
    name = "rathor_tarball",
    image = ":rathor_image",
    repo_tags = ["rathor-nexi:latest"],
)
