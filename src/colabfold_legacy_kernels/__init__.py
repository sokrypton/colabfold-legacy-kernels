"""ColabFold sm_70 and sm_75 CUDA kernels."""

from pathlib import Path

__all__ = [
    "KERNELS",
    "available",
    "kernel_dir",
    "library_path",
    "supported_capabilities",
    "symbol",
]

__version__ = "0.2.0"

_KERNEL_DIR = Path(__file__).resolve().parent / "kernels"

_CAPABILITIES = (70, 75)

# Each kernel lists its libraries from the highest compute capability first.
_TABLE = {
    "attention": (
        (75, "libvolta_mma.so", "VoltaMma"),
        (70, "libvolta_wmma.so", "VoltaWmma"),
    ),
    "layer_norm": (
        (70, "libvolta_ops.so", "VoltaLayerNorm"),
    ),
    # sm_75 only: the wmma (sm_70) attention has no backward yet.
    "attention_bwd": (
        (75, "libvolta_mma_bwd.so", "VoltaMmaBwd"),
    ),
    "gated_dual_proj": (
        (75, "libvolta_ops.so", "VoltaGdp"),
        (70, "libvolta_gdp_wmma.so", "VoltaGdpWmma"),
    ),
}

KERNELS = tuple(_TABLE)
# available() defaults to these, so a wheel built before the backward existed
# and one built after both answer the same question.
_FORWARD_KERNELS = ("attention", "layer_norm", "gated_dual_proj")


def supported_capabilities():
    """Give the compute capabilities that this package contains."""
    return _CAPABILITIES


def _resolve(kernel, cc):
    """Find the library and the FFI symbol for one kernel and one capability."""
    if kernel not in _TABLE:
        raise KeyError(
            "unknown kernel %r, expected one of %s" % (kernel, ", ".join(KERNELS)))
    cc = int(cc)
    if cc not in _CAPABILITIES:
        return cc, None, None
    for min_cc, lib, sym in _TABLE[kernel]:
        if cc >= min_cc:
            return cc, lib, sym
    return cc, None, None


def kernel_dir(cc):
    """Give the directory of the libraries for one capability."""
    cc, _, _ = _resolve("attention", cc)
    return str(_KERNEL_DIR / ("sm%d" % cc))


def library_path(kernel, cc):
    """Give the path of the library that supplies one kernel."""
    cc, lib, _ = _resolve(kernel, cc)
    if lib is None:
        raise FileNotFoundError(
            "no %s kernel for compute capability %d" % (kernel, cc))
    path = _KERNEL_DIR / ("sm%d" % cc) / lib
    if not path.is_file():
        raise FileNotFoundError("%s is not in this package" % path)
    return str(path)


def symbol(kernel, cc):
    """Give the XLA FFI target name that the library exports."""
    cc, lib, sym = _resolve(kernel, cc)
    if sym is None:
        raise FileNotFoundError(
            "no %s kernel for compute capability %d" % (kernel, cc))
    return sym


def available(cc, kernel=None):
    """Tell if the kernels are present. With kernel=None, test all of them."""
    names = _FORWARD_KERNELS if kernel is None else (kernel,)
    for name in names:
        try:
            library_path(name, cc)
        except (FileNotFoundError, KeyError):
            return False
    return True
