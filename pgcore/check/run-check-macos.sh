#!/bin/bash
# Build and run linsolver_check.cpp against the libgimli that
# build-libgimli-macos.sh put in <work>/rebuilt. Needs CXX (the compiler that
# built it) and the runtime environment's prefix (libc++, libomp, OpenBLAS).
#
# usage: run-check-macos.sh <patched gimli tree> <work dir of the build> <rustsolve prefix> <runtime prefix>
# Exits with the check's status (0: every check passes).
set -euo pipefail
GIMLI=$(cd "$1" && pwd); WORK=$(cd "$2" && pwd); PREFIX=$(cd "$3" && pwd); RT=$(cd "$4" && pwd)
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$WORK/check
mkdir -p "$OUT"
: "${CXX:?use the compiler that built libgimli}"
SDK=$(xcrun --show-sdk-path)
export MACOSX_DEPLOYMENT_TARGET=${MACOSX_DEPLOYMENT_TARGET:-13.0}
# libgimli uses -undefined dynamic_lookup on macOS. Its static initializer
# calls log() -> Py_IsInitialized(), even before main and with debug disabled.
# A standalone executable must supply the runtime's Python symbols. Keep the
# dependency even though this check itself calls no Python API; initializing
# Python in main would be too late (and is not needed). conda-forge's
# sysconfig names the static libpython (LDLIBRARY); its libpython package
# installs the shared one as lib/libpython<major.minor><abiflags>.dylib.
PYLIB=$("$RT/bin/python" -c 'import pathlib, sys; print(pathlib.Path(sys.prefix) / "lib" / ("libpython%d.%d%s.dylib" % (sys.version_info[0], sys.version_info[1], getattr(sys, "abiflags", ""))))')
if [ ! -f "$PYLIB" ] || [[ "$PYLIB" != *.dylib ]]; then
    echo "libpython dylib not found: $PYLIB" >&2
    exit 1
fi
"$CXX" -std=gnu++11 -O1 -g -DHAVE_CONFIG_CMAKE_H -Wno-deprecated-declarations \
    -fopenmp -isysroot "$SDK" -mmacosx-version-min="$MACOSX_DEPLOYMENT_TARGET" \
    -I"$GIMLI/core/src" -I"$WORK/build" -I"$PREFIX/include" -I"$RT/include" \
    "$HERE/linsolver_check.cpp" -L"$WORK/rebuilt" -lgimli -L"$RT/lib" \
    -Xlinker -needed_library -Xlinker "$PYLIB" \
    -Wl,-rpath,"$WORK/rebuilt" -Wl,-rpath,"$RT/lib" -o "$OUT/linsolver_check"
otool -L "$OUT/linsolver_check"
DYLD_LIBRARY_PATH="$WORK/rebuilt" "$OUT/linsolver_check"
