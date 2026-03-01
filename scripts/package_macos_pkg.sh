#!/usr/bin/env bash
#
#  package_macos_pkg.sh
#  BeatIt
#
#  Created by Till Toenshoff on 2026-03-01.
#  Copyright © 2026 Till Toenshoff. All rights reserved.
#

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/build}"
DIST_DIR="${DIST_DIR:-$ROOT/dist}"
VERSION="${VERSION:-0.0.0}"
PKG_NAME="${PKG_NAME:-beatit-macos-${VERSION}.pkg}"
PKG_PATH="${PKG_PATH:-$DIST_DIR/$PKG_NAME}"
INSTALLER_SIGN_IDENTITY="${INSTALLER_SIGN_IDENTITY:-}"

PAYLOAD_ROOT="$DIST_DIR/pkgroot"

mkdir -p "$DIST_DIR"
rm -rf "$PAYLOAD_ROOT"

mkdir -p "$PAYLOAD_ROOT/usr/local/bin"
mkdir -p "$PAYLOAD_ROOT/usr/local/lib/beatit"
mkdir -p "$PAYLOAD_ROOT/usr/local/share/beatit/models"
mkdir -p "$PAYLOAD_ROOT/Library/Frameworks"

cp "$BUILD_DIR/beatit" "$PAYLOAD_ROOT/usr/local/bin/beatit"

if [[ -f "$BUILD_DIR/plugins/libbeatit_backend_coreml.dylib" ]]; then
  cp "$BUILD_DIR/plugins/libbeatit_backend_coreml.dylib" \
     "$PAYLOAD_ROOT/usr/local/lib/beatit/libbeatit_backend_coreml.dylib"
fi

if [[ -f "$BUILD_DIR/plugins/libbeatit_backend_torch.dylib" ]]; then
  cp "$BUILD_DIR/plugins/libbeatit_backend_torch.dylib" \
     "$PAYLOAD_ROOT/usr/local/lib/beatit/libbeatit_backend_torch.dylib"
fi

cp -R "$ROOT/models/BeatThis_small0.mlpackage" \
      "$PAYLOAD_ROOT/usr/local/share/beatit/models/BeatThis_small0.mlpackage"

if [[ -f "$ROOT/models/beatthis.pt" ]]; then
  cp "$ROOT/models/beatthis.pt" \
     "$PAYLOAD_ROOT/usr/local/share/beatit/models/beatthis.pt"
fi

cp -R "$BUILD_DIR/BeatIt.framework" "$PAYLOAD_ROOT/Library/Frameworks/BeatIt.framework"

PKGBUILD_ARGS=(
  --root "$PAYLOAD_ROOT"
  --identifier "com.tilltoenshoff.beatit"
  --version "$VERSION"
  --install-location "/"
)

if [[ -n "$INSTALLER_SIGN_IDENTITY" ]]; then
  PKGBUILD_ARGS+=(--sign "$INSTALLER_SIGN_IDENTITY")
fi

pkgbuild "${PKGBUILD_ARGS[@]}" "$PKG_PATH"

echo "Created package: $PKG_PATH"
