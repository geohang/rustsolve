#!/bin/bash
# Build libgimli 1.5.x with rustsolve in place of CHOLMOD/UMFPACK, for an
# existing macOS pgcore from conda (libgimli only, no Python bindings).
#
# Run with a conda-forge C/C++ toolchain environment activated (cxx-compiler:
# conda's clang links the environment's @rpath/libc++.1.dylib, as the original
# libgimli does; Apple's clang would link /usr/lib/libc++). Headers and
# libraries come from the runtime environment that holds pgcore.
#
# usage: build-libgimli-macos.sh <gimli tree> <rustsolve prefix> <work dir> <version> <runtime prefix>
#   <gimli tree>        source at the pgcore's commit, patched by apply_rustsolve.py
#   <rustsolve prefix>  include/rustsolve.h, lib/librustsolve_capi.dylib (install name @rpath/...)
#   <version>           source version for pg.core.versionStr(), e.g. v1.5.3-53-g6241cfab
#   <runtime prefix>    the conda environment with pgcore, boost headers, openblas, llvm-openmp
# MACOSX_DEPLOYMENT_TARGET defaults to 13.0, the original's minimum.
# Result: <work dir>/rebuilt/libgimli.dylib with the original's install name and
# rpaths (@loader_path/, @loader_path/../lib64/), ad-hoc signed, and a copy of
# librustsolve_capi.dylib beside it.
set -euo pipefail
GIMLI=$(cd "$1" && pwd); PREFIX=$(cd "$2" && pwd); WORK=$3; VERSION=$4; RT=$(cd "$5" && pwd)
mkdir -p "$WORK"; WORK=$(cd "$WORK" && pwd)
export MACOSX_DEPLOYMENT_TARGET=${MACOSX_DEPLOYMENT_TARGET:-13.0}
SDK=$(xcrun --show-sdk-path)
: "${CC:?activate a conda-forge compiler environment (CC unset)}"
: "${CXX:?activate a conda-forge compiler environment (CXX unset)}"
PY=$RT/bin/python

# Triangle (Shewchuk), linked statically into libgimli: the trilibrary
# target, without -DLINUX on macOS
TRI_URL=https://www.netlib.org/voronoi/triangle.zip
TRI_SHA256=1766327add038495fa3499e9b7cc642179229750f7201b94f8e1b7bee76f8480
if [ ! -f "$WORK/tri/lib/libtriangle.a" ]; then
    mkdir -p "$WORK/dl" "$WORK/trisrc" "$WORK/tri/lib" "$WORK/tri/include"
    [ -f "$WORK/dl/triangle.zip" ] || curl -fsSL -o "$WORK/dl/triangle.zip" "$TRI_URL"
    echo "$TRI_SHA256  $WORK/dl/triangle.zip" | shasum -a 256 -c -
    unzip -oq "$WORK/dl/triangle.zip" -d "$WORK/trisrc"
    # K&R definitions: newer C defaults (C23) reject them
    (cd "$WORK/trisrc" && "$CC" -std=gnu17 -w -fPIC -O -DTRILIBRARY -isysroot "$SDK" \
        -c triangle.c -o triangle.o && ar rcs "$WORK/tri/lib/libtriangle.a" triangle.o)
    cp "$WORK/trisrc/triangle.h" "$WORK/tri/include/"
fi

# gimli's CMake insists on the binding generator (castxml, pygccxml,
# pyplusplus) before it configures libgimli. Placeholders satisfy the checks;
# the `gimli` target uses none of them.
NB=$WORK/nobind
PYV=$("$PY" -c "import sys; print(f'{sys.version_info[0]}{sys.version_info[1]}')")
mkdir -p "$NB/bin" "$NB/pygccxml" "$NB/pyplusplus"
cp /usr/bin/true "$NB/bin/castxml"
cp /usr/bin/true "$NB/bin/clang"
: > "$NB/pygccxml/__init__.py"; : > "$NB/pyplusplus/__init__.py"

rm -rf "$WORK/build"; mkdir -p "$WORK/build"; cd "$WORK/build"
cmake -G "Unix Makefiles" "$GIMLI" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" \
    -DCMAKE_OSX_DEPLOYMENT_TARGET="$MACOSX_DEPLOYMENT_TARGET" -DCMAKE_OSX_SYSROOT="$SDK" \
    -DCMAKE_PREFIX_PATH="$RT" -DCMAKE_FIND_FRAMEWORK=LAST \
    -DPython_EXECUTABLE="$PY" -DPython_ROOT_DIR="$RT" \
    -DBoost_INCLUDE_DIR="$RT/include" \
    -DTriangle_PREFIX_PATH="$WORK/tri" -DRUSTSOLVE_PREFIX="$PREFIX" -DGIMLI_VERSION="$VERSION" \
    -DAVOID_CPPUNIT=1 -DTHIRDPARTY_DIR="$WORK/thirdParty" -DSKIP_SPHINX=1 \
    -DNOREADPROC=1 \
    -DCASTXML_EXECUTABLE="$NB/bin/castxml" -DCLANG_EXECUTABLE="$NB/bin/clang" \
    -DPYGCCXML_PATH="$NB" -DPYPLUSPLUS_PATH="$NB" \
    -DBoost_PYTHON_FOUND=TRUE -DBoost_PYTHON_LIBRARY="$RT/lib/libboost_python$PYV.dylib" \
    -DOpenMP_C_FLAGS="-fopenmp -I$RT/include" -DOpenMP_C_LIB_NAMES=omp \
    -DOpenMP_CXX_FLAGS="-fopenmp -I$RT/include" -DOpenMP_CXX_LIB_NAMES=omp \
    -DOpenMP_omp_LIBRARY="$RT/lib/libomp.dylib" \
    > configure.log 2>&1 || { tail -60 configure.log; exit 1; }
grep -E "rustsolve:|CHOLMOD_LIBRARIES|UMFPACK_LIBRARIES|TRIANGLE_FOUND|BLAS_LIBRARIES|OpenMP" configure.log || true
# the original links libomp; a silent miss here would change the library
grep -q "Found OpenMP_CXX" configure.log || { echo "OpenMP not found"; grep -n -i openmp configure.log; exit 1; }
grep -E "PACKAGE_VERSION|CHOLMOD_FOUND|UMFPACK_FOUND" config.cmake.h
# CMake 4's find modules no longer set every upper-case <NAME>_FOUND the
# template reads (readproc on macOS: hence NOREADPROC); an empty one breaks
# an #if in the sources. (The version numbers and HAVE_* flags are empty in
# every build and only tested with #ifdef.)
if grep -nE "^#define [A-Za-z_]*(_FOUND|USE_IPC|CONDA_BUILD)[[:space:]]*$" config.cmake.h; then
    echo "config.cmake.h leaves a value tested by #if empty"; exit 1
fi
make gimli -j"$(sysctl -n hw.ncpu)" > build.log 2>&1 || { grep -E "error" build.log | head -30; tail -30 build.log; exit 1; }

LIB=$(find "$WORK/build" -name libgimli.dylib -type f | head -1)
OUT=$WORK/rebuilt
rm -rf "$OUT"; mkdir -p "$OUT"
cp "$LIB" "$OUT/libgimli.dylib"
cp "$PREFIX/lib/librustsolve_capi.dylib" "$OUT/"
L=$OUT/libgimli.dylib
# the original's identity and rpaths, so the file drops in where it was
install_name_tool -id @rpath/libgimli.dylib "$L"
for r in $(otool -l "$L" | awk '/cmd LC_RPATH/{getline; getline; print $2}'); do
    install_name_tool -delete_rpath "$r" "$L"
done
install_name_tool -add_rpath @loader_path/ -add_rpath @loader_path/../lib64/ "$L"
codesign --force -s - "$L"
otool -D "$L"; otool -L "$L"
otool -l "$L" | grep -A2 -E "LC_RPATH|LC_BUILD_VERSION" | grep -E "path|minos|sdk"
# the bindings load the environment's libc++ and libomp: a system or absolute
# copy here would put a second C++ runtime in the process
if otool -L "$L" | tail -n +2 | grep -E "libc\+\+|libomp|libopenblas" | grep -v "@rpath/"; then
    echo "libgimli links a C++/OpenMP/BLAS library outside @rpath"; exit 1
fi
otool -L "$L" | grep -q "@rpath/librustsolve_capi.dylib" || { echo "librustsolve_capi not linked by @rpath"; exit 1; }
