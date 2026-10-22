#!/bin/bash
# Build and run linsolver_check.cpp against a libgimli built by
# build-libgimli-msys2.sh, in the MSYS2 MINGW64 shell.
#
# usage: run-check-msys2.sh <patched gimli tree> <work dir of the build> <rustsolve prefix>
# Exits with the check's status (0: every check passes).
set -euo pipefail
GIMLI=$(cd "$1" && pwd); WORK=$(cd "$2" && pwd); PREFIX=$(cd "$3" && pwd)
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$WORK/check
mkdir -p "$OUT"
g++ -std=gnu++11 -O1 -DHAVE_CONFIG_CMAKE_H -Wno-deprecated-declarations \
    -I"$GIMLI/core/src" -I"$WORK/build" -I"$PREFIX/include" \
    "$HERE/linsolver_check.cpp" -L"$WORK/build/lib" -lgimli -o "$OUT/linsolver_check.exe"
PATH="$WORK/build/bin:$PREFIX/bin:$PATH" "$OUT/linsolver_check.exe"
