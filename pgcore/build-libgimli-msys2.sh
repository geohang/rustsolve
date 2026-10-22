#!/bin/bash
# Build libgimli 1.5.x with rustsolve in place of CHOLMOD/UMFPACK, for an
# existing pgcore of the same version (libgimli only, no Python bindings).
# Run in the MSYS2 MINGW64 shell (msvcrt, like the pgcore 1.5.0 build), with:
#   pacman -S --needed mingw-w64-x86_64-{gcc,cmake,make,boost,openblas} make patch unzip
#
# usage: build-libgimli-msys2.sh <gimli tree> <rustsolve prefix> <work dir> <version> <python>
#   <gimli tree>        source at the pgcore's commit, patched by apply_rustsolve.py
#   <rustsolve prefix>  include/rustsolve.h, lib/librustsolve_capi.dll.a, bin/rustsolve_capi.dll
#   <version>           source version for pg.core.versionStr(), e.g. v1.5.0-1-g5991b6ac
#   <python>            a Python with NumPy; CMake requires it although nothing here uses it
# Result: <work dir>/build/bin/libgimli.dll
set -euo pipefail
GIMLI=$(cd "$1" && pwd); PREFIX=$(cd "$2" && pwd); WORK=$3; VERSION=$4; PY=$5
mkdir -p "$WORK"; WORK=$(cd "$WORK" && pwd)

# Triangle (Shewchuk), statically into libgimli as gimli's own build does
TRI_URL=https://www.netlib.org/voronoi/triangle.zip
TRI_SHA256=1766327add038495fa3499e9b7cc642179229750f7201b94f8e1b7bee76f8480
if [ ! -f "$WORK/tri/lib/libtriangle.a" ]; then
    mkdir -p "$WORK/dl" "$WORK/trisrc" "$WORK/tri/lib" "$WORK/tri/include"
    [ -f "$WORK/dl/triangle.zip" ] || curl -fsSL -o "$WORK/dl/triangle.zip" "$TRI_URL"
    echo "$TRI_SHA256  $WORK/dl/triangle.zip" | sha256sum -c -
    unzip -oq "$WORK/dl/triangle.zip" -d "$WORK/trisrc"
    (cd "$WORK/trisrc" && patch -p1 < "$GIMLI/core/scripts/patches/triangle-mingw-win64.patch")
    # K&R definitions: GCC 15+ defaults to C23, which rejects them
    (cd "$WORK/trisrc" && gcc -std=gnu17 -fPIC -O -DCPU86 -DTRILIBRARY -c triangle.c -o triangle.o \
        && ar rcs "$WORK/tri/lib/libtriangle.a" triangle.o)
    cp "$WORK/trisrc/triangle.h" "$WORK/tri/include/"
fi

# gimli's CMake insists on the binding generator (castxml, pygccxml,
# pyplusplus) and on boost_python before it configures libgimli. Placeholders
# satisfy the checks; the `gimli` target uses none of them.
NB=$WORK/nobind
PYV=$("$PY" -c "import sys; print(f'{sys.version_info[0]}{sys.version_info[1]}')")
mkdir -p "$NB/bin" "$NB/lib" "$NB/pygccxml" "$NB/pyplusplus"
cp /usr/bin/true.exe "$NB/bin/castxml.exe"
cp /usr/bin/true.exe "$NB/bin/clang.exe"
: > "$NB/pygccxml/__init__.py"; : > "$NB/pyplusplus/__init__.py"
: > "$NB/lib/libboost_python$PYV-mt-x64.dll"; : > "$NB/lib/libboost_python$PYV-mt-x64.dll.a"

PYROOT=$(dirname "$PY")
rm -rf "$WORK/build"; mkdir -p "$WORK/build"; cd "$WORK/build"
cmake -G "Unix Makefiles" "$GIMLI" -DCMAKE_BUILD_TYPE=Release \
    -DPython_EXECUTABLE="$PY" -DPython_ROOT_DIR="$PYROOT" \
    -DBoost_INCLUDE_DIR=/mingw64/include \
    -DTriangle_PREFIX_PATH="$WORK/tri" -DRUSTSOLVE_PREFIX="$PREFIX" -DGIMLI_VERSION="$VERSION" \
    -DAVOID_CPPUNIT=1 -DTHIRDPARTY_DIR="$WORK/thirdParty" -DSKIP_SPHINX=1 \
    -DCASTXML_EXECUTABLE="$NB/bin/castxml.exe" -DCLANG_EXECUTABLE="$NB/bin/clang.exe" \
    -DPYGCCXML_PATH="$NB" -DPYPLUSPLUS_PATH="$NB" \
    -DBoost_PYTHON_FOUND=TRUE -DBoost_PYTHON_LIBRARY="$NB/lib/libboost_python$PYV-mt-x64.dll.a" \
    > configure.log 2>&1 || { tail -40 configure.log; exit 1; }
grep -E "rustsolve:|CHOLMOD_LIBRARIES|UMFPACK_LIBRARIES|TRIANGLE_FOUND" configure.log
grep -E "PACKAGE_VERSION|CHOLMOD_FOUND|UMFPACK_FOUND" config.cmake.h
make gimli -j"$(nproc)" > build.log 2>&1 || { grep -E "error" build.log | head -20; exit 1; }
ls -la bin/libgimli.dll
