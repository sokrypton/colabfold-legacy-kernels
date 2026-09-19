"""VoltaMmaFwd / VoltaMmaBwd against JAX's own autodiff.

  KERNEL_DIR=<dir with libvolta_mma*.so> python tests/test_attn_bwd.py

The kernels are sm_75+, so this runs on any Turing-or-newer card once the
libraries are built for it -- which is how it can be developed on an Ampere
box and only confirmed on the T4 it is meant for.
"""
import ctypes
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

KERNEL_DIR = os.environ.get('KERNEL_DIR', '.')
NEG = -1.0e4


def register():
  for lib, syms in (('libvolta_mma.so', ('VoltaMma', 'VoltaMmaFwd')),
                    ('libvolta_mma_bwd.so', ('VoltaMmaBwd',))):
    handle = ctypes.cdll.LoadLibrary(os.path.join(KERNEL_DIR, lib))
    for sym in syms:
      jax.ffi.register_ffi_target(
          sym, jax.ffi.pycapsule(getattr(handle, sym)), platform='CUDA')


def fwd(q, k, v, bias, kmask, scale, bq=64, bk=32):
  n, h, sq, d = q.shape
  sk = k.shape[2]
  return jax.ffi.ffi_call(
      'VoltaMmaFwd',
      (jax.ShapeDtypeStruct((n, h, sq, d), jnp.float16),
       jax.ShapeDtypeStruct((n, h, sq), jnp.float32)),
      vmap_method='sequential')(
          q, k, v, bias, kmask, scale=np.float32(scale),
          block_q=np.int64(bq), block_k=np.int64(bk))


def bwd(q, k, v, bias, kmask, dout, lse, delta, scale, bq=64, bk=32):
  n, h, sq, d = q.shape
  sk = k.shape[2]
  return jax.ffi.ffi_call(
      'VoltaMmaBwd',
      (jax.ShapeDtypeStruct(q.shape, jnp.float16),
       jax.ShapeDtypeStruct(k.shape, jnp.float16),
       jax.ShapeDtypeStruct(v.shape, jnp.float16),
       jax.ShapeDtypeStruct((h, sq, sk), jnp.float32)),
      vmap_method='sequential')(
          q, k, v, bias, kmask, dout, lse, delta, scale=np.float32(scale),
          block_q=np.int64(bq), block_k=np.int64(bk))


def reference(q, k, v, bias, kmask, scale):
  """fp32 attention with the kernel's own masking convention."""
  logits = scale * jnp.einsum('nhqd,nhkd->nhqk', q, k) + bias[None]
  logits = jnp.where(kmask[:, None, None, :] != 0, logits, NEG)
  p = jax.nn.softmax(logits, axis=-1)
  return jnp.einsum('nhqk,nhkd->nhqd', p, v)


def rel(a, b):
  a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
  denom = max(np.abs(b).max(), 1e-6)
  return float(np.abs(a - b).max() / denom)


def run(n=3, h=4, sq=96, sk=96, d=32, seed=0, bq=64, bk=32):
  key = jax.random.PRNGKey(seed)
  ks = jax.random.split(key, 6)
  f16 = lambda x: x.astype(jnp.float16)
  q = f16(jax.random.normal(ks[0], (n, h, sq, d)) * 0.5)
  k = f16(jax.random.normal(ks[1], (n, h, sk, d)) * 0.5)
  v = f16(jax.random.normal(ks[2], (n, h, sk, d)) * 0.5)
  bias = f16(jax.random.normal(ks[3], (h, sq, sk)) * 0.5)
  kmask = (jax.random.uniform(ks[4], (n, sk)) > 0.15).astype(jnp.uint8)
  dout = f16(jax.random.normal(ks[5], (n, h, sq, d)) * 0.5)
  scale = float(d) ** -0.5

  qf, kf, vf, bf = [x.astype(jnp.float32) for x in (q, k, v, bias)]
  ref_out, vjp = jax.vjp(
      lambda a, b, c, e: reference(a, b, c, e, kmask, scale), qf, kf, vf, bf)
  ref_dq, ref_dk, ref_dv, ref_dbias = vjp(dout.astype(jnp.float32))

  out, lse = fwd(q, k, v, bias, kmask, scale, bq, bk)
  delta = jnp.sum(out.astype(jnp.float32) * dout.astype(jnp.float32), -1)
  dq, dk, dv, dbias = bwd(q, k, v, bias, kmask, dout, lse, delta, scale, bq, bk)

  # lse comes back in the log2 domain, which is what the kernel's exp2 wants.
  ref_logits = scale * jnp.einsum('nhqd,nhkd->nhqk', qf, kf) + bf[None]
  ref_logits = jnp.where(kmask[:, None, None, :] != 0, ref_logits, NEG)
  ref_lse2 = jax.scipy.special.logsumexp(ref_logits, -1) * 1.4426950408889634

  rows = [('out', out, ref_out), ('lse2', lse, ref_lse2), ('dq', dq, ref_dq),
          ('dk', dk, ref_dk), ('dv', dv, ref_dv), ('dbias', dbias, ref_dbias)]
  worst = 0.0
  for name, got, want in rows:
    r = rel(got, want)
    worst = max(worst, r)
    print(f'  {name:6s} rel {r:.2e}   max|ref| {np.abs(np.asarray(want)).max():.3f}')
  return worst


if __name__ == '__main__':
  register()
  bad = 0
  for kwargs in (dict(sq=96, sk=96, d=32),
                 dict(sq=96, sk=96, d=32, bq=64, bk=64),
                 dict(sq=64, sk=64, d=64, bk=64),   # the mma forward has no (64, 64, 32)
                 dict(sq=70, sk=83, d=16),        # ragged, both axes
                 dict(sq=128, sk=32, d=8, n=1, h=1)):
    print(kwargs)
    worst = run(**kwargs)
    ok = worst < 0.05
    bad += not ok
    print('  ->', 'OK' if ok else 'FAIL', f'(worst {worst:.2e})')
  print('RESULT:', 'PASS' if not bad else f'FAIL ({bad})')
  sys.exit(0 if not bad else 1)
