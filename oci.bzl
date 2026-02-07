# oci.bzl – OCI multi-stage image definitions for Rathor-NEXi

load("@rules_oci//oci:defs.bzl", "oci_image", "oci_tarball")
load("@rules_pkg//:pkg.bzl", "pkg_tar")
load("@rules_vite//vite:defs.bzl", "vite")

# ─── Stage 1: Vite build artifact (hermetic via Bazel) ────────────────
vite(
    name = "vite_build",
    srcs = glob([
        "src/**",
        "public/**",
        "*.json",
        "*.ts",
        "*.tsx",
        "*.js",
        "vite.config.ts",
    ]),
    entry_point = "src/main.tsx",
    vite_config = "vite.config.ts",
    data = [
        "//:node_modules",
        "//:tsconfig.json",
        "//:package.json",
    ],
    env = {"NODE_ENV": "production"},
    args = ["build"],
    outs = ["dist"],
)

pkg_tar(
    name = "vite_dist_tar",
    srcs = [":vite_build"],
    package_dir = "/usr/share/nginx/html",
    mode = "0755",
)

# ─── Stage 2: Runtime image (nginx + dist only, distroless-like) ───────
oci_image(
    name = "rathor_runtime",
    base = "@nginx_distroless",  # minimal nginx + no shell (distroless variant)
    tars = [":vite_dist_tar"],
    entrypoint = ["/usr/sbin/nginx"],
    cmd = ["-g", "daemon off;"],
    exposed_ports = ["80/tcp"],
    labels = {
        "org.opencontainers.image.source": "https://github.com/Eternally-Thriving-Grandmasterism/Rathor-NEXi",
        "org.opencontainers.image.description": "Rathor NEXi – Sovereign offline AGI lattice",
        "org.opencontainers.image.licenses": "MIT",
        "org.opencontainers.image.vendor": "Eternally-Thriving-Grandmasterism",
    },
)

# Multi-platform tarball (amd64 + arm64)
oci_tarball(
    name = "rathor_multiarch_tarball",
    image = ":rathor_runtime",
    repo_tags = ["rathor-nexi:latest"],
    architecture = ["amd64", "arm64"],
)

# Convenience alias
alias(
    name = "image",
    actual = ":rathor_runtime",
)
