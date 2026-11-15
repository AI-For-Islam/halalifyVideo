#!/usr/bin/env bash
# Builds a release archive for the platform it runs on:
#   Windows (Git Bash):  dist/halalify-<version>-windows-x86_64.zip
#   Linux:               dist/halalify-<version>-linux-x86_64.tar.gz
# Each archive unpacks to a self-contained folder with the program, a
# launcher, the documentation and an empty "data" folder that the program
# installs its runtime and models into. The Windows archive also carries the
# desktop app, HalalifyVid.exe (see desktop/).
set -euo pipefail
cd "$(dirname "$0")/.."

version=$(grep -m1 '^version' Cargo.toml | cut -d'"' -f2)
case "$(uname -s)" in
    MINGW* | MSYS* | CYGWIN*) os=windows; exe=halalify.exe; launcher=HalalifyVid.cmd ;;
    Linux) os=linux; exe=halalify; launcher=halalifyvid.sh ;;
    *) echo "unsupported system: $(uname -s)" >&2; exit 1 ;;
esac
target_dir=${CARGO_TARGET_DIR:-target}

# The compiler records source paths in the binary (for panic messages). Map
# the local cargo home and the checkout to neutral names, so a release
# carries no user name or folder layout of the machine that built it.
cargo_home=${CARGO_HOME:-$HOME/.cargo}
project=$PWD
if [ "$os" = windows ]; then
    cargo_home=$(cygpath -w "$cargo_home")
    project=$(cygpath -w "$project")
fi
flags=("--remap-path-prefix=$cargo_home=cargo" "--remap-path-prefix=$project=halalify")
# These flags replace .cargo/config.toml's, so the static C runtime is repeated.
[ "$os" = windows ] && flags+=("-Ctarget-feature=+crt-static")
# The encoded form (separated by 0x1f) keeps paths with spaces intact.
CARGO_ENCODED_RUSTFLAGS=$(IFS=$'\x1f'; echo "${flags[*]}")
export CARGO_ENCODED_RUSTFLAGS

cargo build --release --locked
if [ "$os" = windows ]; then
    cargo build --release --locked --manifest-path desktop/src-tauri/Cargo.toml
fi

name="halalify-$version-$os-x86_64"
stage="dist/$name"
rm -rf "$stage"
mkdir -p "$stage/data" "$stage/docs"
cp "$target_dir/release/$exe" "$stage/"
cp "packaging/$launcher" "$stage/"
if [ "$os" = windows ]; then
    cp "${CARGO_TARGET_DIR:-desktop/src-tauri/target}/release/HalalifyVid.exe" "$stage/"
fi
cp README.md LICENSE "$stage/"
cp docs/DESIGN.md docs/RELEASES.md "$stage/docs/"
cp packaging/data-README.txt "$stage/data/README.txt"

cd dist
if [ "$os" = windows ]; then
    rm -f "$name.zip"
    # Windows' own bsdtar writes zips with forward slashes, which every
    # unzip tool understands (PowerShell's Compress-Archive does not).
    "${SYSTEMROOT:-/c/Windows}/System32/tar.exe" -a -cf "$name.zip" "$name"
    archive="$name.zip"
else
    chmod +x "$name/$exe" "$name/$launcher"
    tar -czf "$name.tar.gz" "$name"
    archive="$name.tar.gz"
fi
sha256sum "$archive" > "$archive.sha256"
echo "built dist/$archive ($(du -h "$archive" | cut -f1))"
