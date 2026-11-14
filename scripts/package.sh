#!/usr/bin/env bash
# Builds a release archive for the platform it runs on:
#   Windows (Git Bash):  dist/halalify-<version>-windows-x86_64.zip
#   Linux:               dist/halalify-<version>-linux-x86_64.tar.gz
# Each archive unpacks to a self-contained folder with the program, a
# launcher, the documentation and an empty "data" folder that the program
# installs its runtime and models into.
set -euo pipefail
cd "$(dirname "$0")/.."

version=$(grep -m1 '^version' Cargo.toml | cut -d'"' -f2)
case "$(uname -s)" in
    MINGW* | MSYS* | CYGWIN*) os=windows; exe=halalify.exe; launcher=HalalifyVid.cmd ;;
    Linux) os=linux; exe=halalify; launcher=halalifyvid.sh ;;
    *) echo "unsupported system: $(uname -s)" >&2; exit 1 ;;
esac
target_dir=${CARGO_TARGET_DIR:-target}

cargo build --release --locked

name="halalify-$version-$os-x86_64"
stage="dist/$name"
rm -rf "$stage"
mkdir -p "$stage/data" "$stage/docs"
cp "$target_dir/release/$exe" "$stage/"
cp "packaging/$launcher" "$stage/"
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
