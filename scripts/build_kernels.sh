#!/usr/bin/env bash
# Build the kernels for one compute capability. ARCH=70 needs CUDA 12 nvcc.
set -euo pipefail

OUT="${1:-src/colabfold_legacy_kernels/kernels}"
ARCH="${ARCH:-75}"
PY="${PYTHON:-python3}"

here="$(pwd)"
repo="$(cd "$(dirname "$0")/.." && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

# CUDA 13 does not support Volta. sm_70 needs a CUDA 12 toolkit.
find_nvcc() {
  if [ -n "${NVCC:-}" ] && [ -x "${NVCC}" ]; then echo "$NVCC"; return; fi
  if [ "$ARCH" -lt 75 ]; then
    for c in /usr/local/cuda-12.*/bin/nvcc /usr/local/cuda-12/bin/nvcc; do
      [ -x "$c" ] && { echo "$c"; return; }
    done
  fi
  command -v nvcc 2>/dev/null || true
}
NVCC_BIN="$(find_nvcc)"
if [ -z "$NVCC_BIN" ]; then
  echo "!! no nvcc found; set \$NVCC" >&2
  exit 1
fi
CUDA_MAJOR="$("$NVCC_BIN" --version | sed -n 's/.*release \([0-9]*\)\..*/\1/p')"
CUOBJDUMP="${CUOBJDUMP:-$(dirname "$NVCC_BIN")/cuobjdump}"
if [ ! -x "$CUOBJDUMP" ]; then
  echo "!! no cuobjdump next to $NVCC_BIN; set \$CUOBJDUMP" >&2
  exit 1
fi

# nvdisasm cannot read sm_70, thus Volta gets no instruction check.
USE_SASS=1
[ "$ARCH" -lt 75 ] && USE_SASS=0
if [ "$ARCH" -lt 75 ] && [ "$CUDA_MAJOR" != "12" ]; then
  echo "!! sm_${ARCH} needs a CUDA 12 nvcc (got CUDA $CUDA_MAJOR; 13 dropped Volta)" >&2
  exit 1
fi

FFI_INC="${FFI_INCLUDE:-}"
if [ -z "$FFI_INC" ]; then
  FFI_INC="$("$PY" -c 'import jax; print(jax.ffi.include_dir())')"
fi
if [ ! -f "$FFI_INC/xla/ffi/api/ffi.h" ]; then
  echo "!! no xla/ffi/api/ffi.h under $FFI_INC (set \$FFI_INCLUDE or pip install jax)" >&2
  exit 1
fi

# A handler runs on its own XLA_FFI_API and newer, so newer headers raise the jax floor.
FFI_API="$(sed -n 's/^#define XLA_FFI_API_\(MAJOR\|MINOR\) *//p' "$FFI_INC/xla/ffi/api/c_api.h" | paste -sd. -)"
if [ "$FFI_API" != "0.1" ] && [ -z "${ALLOW_NEW_FFI:-}" ]; then
  echo "!! XLA_FFI_API $FFI_API headers: point \$FFI_INCLUDE at a jax 0.6 wheel, or ALLOW_NEW_FFI=1" >&2
  exit 1
fi

CUTLASS="${CUTLASS_DIR:-}"
if [ -z "$CUTLASS" ]; then
  CUTLASS="$(bash "$repo/scripts/fetch_cutlass.sh" "$work/cutlass")"
fi
if [ ! -d "$CUTLASS/include" ]; then
  echo "!! no include/ under CUTLASS_DIR=$CUTLASS" >&2
  exit 1
fi

case "$OUT" in /*) dest="$OUT/sm${ARCH}" ;; *) dest="$here/$OUT/sm${ARCH}" ;; esac
mkdir -p "$dest"

echo "nvcc     $NVCC_BIN (CUDA $CUDA_MAJOR)"
echo "ffi      $FFI_INC (api $FFI_API)"
echo "cutlass  $CUTLASS"
echo "out      $dest"

# Static link libstdc++
STATIC=""
if [ -f "$(${CXX:-c++} -print-file-name=libstdc++.a 2>/dev/null)" ]; then
  STATIC="-Xcompiler -static-libstdc++ -Xcompiler -static-libgcc"
else
  echo "error: no libstdc++.a"
  exit 1
fi

FLAGS="-O3 -std=c++17 -Xcompiler -fPIC $STATIC
       -gencode arch=compute_${ARCH},code=sm_${ARCH}
       -Wno-deprecated-gpu-targets -diag-suppress 940,2473
       -I$FFI_INC -I$CUTLASS/include"

run_nvcc() {
  if ! "$NVCC_BIN" $FLAGS "$@" >"$work/nvcc.log" 2>&1; then
    cat "$work/nvcc.log" >&2
    return 1
  fi
}

# Fields: library : source : minimum arch : arch that must use tensor cores.
# VoltaLayerNorm uses no tensor cores, thus libvolta_ops.so needs none below sm_75.
UNITS="
libvolta_wmma.so:volta_wmma_attn.cu:70:70
libvolta_gdp_wmma.so:volta_gdp_wmma.cu:70:70
libvolta_ops.so:volta_ops.cu:70:75
libvolta_mma.so:volta_mma_attn.cu:75:75
libvolta_mma_bwd.so:volta_mma_attn_bwd.cu:75:75
"

built=0
for u in $UNITS; do
  IFS=: read -r so src minarch tcarch <<EOU
$u
EOU
  if [ "$ARCH" -lt "$minarch" ]; then
    echo "skip $so (needs sm_${minarch}+)"
    continue
  fi
  if ! run_nvcc -shared -o "$dest/$so" "$repo/cuda/$src"; then
    echo "!! nvcc failed to build $so for sm_${ARCH}" >&2
    exit 1
  fi
  if [ "$USE_SASS" = "1" ]; then
    sass="$("$CUOBJDUMP" --dump-sass "$dest/$so" 2>/dev/null || true)"
    if [ -z "$sass" ]; then
      echo "!! cuobjdump gave no SASS for $so on sm_${ARCH}" >&2
      exit 1
    fi
    mma="$(printf '%s' "$sass" | grep -cE 'HMMA|OMMA' || true)"
    brk="$(printf '%s' "$sass" | grep -cE '\bBPT\b' || true)"
  else
    mma=-1
    brk=0
  fi

  strip "$dest/$so" 2>/dev/null || true

  # The library must contain the code for this arch only.
  elves="$("$CUOBJDUMP" --list-elf "$dest/$so" | awk '{print $NF}')"
  if [ -z "$elves" ] || echo "$elves" | grep -qv "sm_${ARCH}\.cubin$"; then
    echo "!! $so contains cubins that are not sm_${ARCH}:" >&2
    echo "$elves" >&2
    exit 1
  fi

  if [ "$brk" -gt 0 ]; then
    echo "!! $so: $brk brkpt traps -- CUTLASS mma compiled but not ENABLED for sm_${ARCH}" >&2
    exit 1
  fi
  if [ "$mma" -ge 0 ] && [ "$ARCH" -ge "$tcarch" ] && [ "$mma" -eq 0 ]; then
    echo "!! $so: no tensor-core instructions in the code for sm_${ARCH}" >&2
    exit 1
  fi
  if [ "$mma" -ge 0 ]; then
    echo "ok       sm${ARCH}/$so ($mma tensor-core ops)"
  else
    echo "ok       sm${ARCH}/$so"
  fi
  built=$((built + 1))
done

if [ "$built" -eq 0 ]; then
  echo "!! nothing built for sm_${ARCH}" >&2
  exit 1
fi
echo "built $built libraries into $dest"
