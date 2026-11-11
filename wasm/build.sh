#!/bin/sh
# ブラウザのデモ（docs/）用の WASM をビルドする。EMSDK で emsdk の場所を変えられる。
#   sh wasm/build.sh            → docs/bdet.js, docs/bdet.wasm
set -e
cd "$(dirname "$0")/.."
EMSDK="${EMSDK:-/c/prog/emsdk/emsdk}"
EMCC="$EMSDK/upstream/emscripten/emcc.exe"
[ -x "$EMCC" ] || EMCC="$EMSDK/upstream/emscripten/emcc"
"$EMCC" -std=c++20 -O3 -msimd128 -DNDEBUG -DUSE_EIGEN -Ipure -Ipure/third_party -Ipure/third_party/eigen_flat \
  -s MODULARIZE=1 -s EXPORT_NAME=createBdet -s ALLOW_MEMORY_GROWTH=1 -s INITIAL_MEMORY=268435456 -s STACK_SIZE=16777216 \
  -s EXPORTED_RUNTIME_METHODS='["HEAPU8","HEAPF32"]' -s EXPORTED_FUNCTIONS='["_malloc","_free"]' \
  wasm/bdet_wasm.cpp -o docs/bdet.js
echo "built docs/bdet.js"
