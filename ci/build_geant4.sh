#!/usr/bin/env bash
# Static Geant4 for the lighthit-g4 wheel.
#
#   ci/build_geant4.sh PREFIX
#
# Downloads the Geant4 source release, checks it against the SHA-256 below and
# installs static, position-independent libraries into PREFIX.  Data sets are
# not installed: the package finds or downloads them at run time.
set -euo pipefail

PREFIX=${1:?usage: build_geant4.sh PREFIX}
VERSION=11.4.1
SHA256=550111022a72241411ec236265f02eb564302c5e9e45961399be36774da24e6e
URL=https://github.com/Geant4/geant4/archive/refs/tags/v${VERSION}.tar.gz
JOBS=${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu)}

if [ -f "${PREFIX}/lib/cmake/Geant4/Geant4Config.cmake" ] || [ -f "${PREFIX}/lib64/cmake/Geant4/Geant4Config.cmake" ]; then
  echo "Geant4 already installed in ${PREFIX}"
  exit 0
fi

WORK=$(mktemp -d)
trap 'rm -rf "${WORK}"' EXIT
# CMake and Ninja from PyPI when the machine has none
if ! command -v cmake >/dev/null || ! command -v ninja >/dev/null; then
  PY=$(command -v python3 || echo /opt/python/cp312-cp312/bin/python)
  "${PY}" -m venv "${WORK}/tools"
  "${WORK}/tools/bin/python" -m pip install --quiet cmake ninja
  export PATH="${WORK}/tools/bin:${PATH}"
fi
curl -fsSL -o "${WORK}/geant4.tar.gz" "${URL}"
if command -v sha256sum >/dev/null; then
  echo "${SHA256}  ${WORK}/geant4.tar.gz" | sha256sum -c -
else
  echo "${SHA256}  ${WORK}/geant4.tar.gz" | shasum -a 256 -c -
fi
mkdir "${WORK}/src"
tar xzf "${WORK}/geant4.tar.gz" -C "${WORK}/src" --strip-components 1

cmake -S "${WORK}/src" -B "${WORK}/build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
  -DCMAKE_INSTALL_LIBDIR=lib \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DBUILD_SHARED_LIBS=OFF \
  -DBUILD_STATIC_LIBS=ON \
  -DGEANT4_BUILD_MULTITHREADED=ON \
  -DGEANT4_BUILD_TLS_MODEL=global-dynamic \
  -DGEANT4_USE_SYSTEM_EXPAT=OFF \
  -DGEANT4_USE_GDML=OFF \
  -DGEANT4_INSTALL_DATA=OFF \
  -DGEANT4_INSTALL_EXAMPLES=OFF
cmake --build "${WORK}/build" --parallel "${JOBS}"
cmake --install "${WORK}/build"
