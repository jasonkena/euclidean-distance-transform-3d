"""
License: GNU 3.0

Expand Parabola
Based off of William Silversmith's https://github.com/seung-lab/euclidean-distance-transform-3d
Jason Ken Adhinarta, 2022
"""
from libc.stdint cimport (
  uint8_t, uint16_t, uint32_t, uint64_t,
   int8_t,  int16_t,  int32_t,  int64_t
)
from libcpp cimport bool as native_bool

import multiprocessing

from cpython cimport array
cimport numpy as np
import numpy as np

__VERSION__ = '0.1'

cdef extern from "expand_parabola.hpp" namespace "py_expand_parabola":
  cdef void _expand_3d_edt[T](
    T* core_squared,
    size_t sx, size_t sy, size_t sz,
    float wx, float wy, float wz,
    int parallel,
  ) nogil

def expand_edt(
    data, anisotropy=(1.0, 1.0, 1.0),
    order='C',
    int parallel=1
  ):

  cdef size_t sx = data.shape[2]
  cdef size_t sy = data.shape[1]
  cdef size_t sz = data.shape[0]
  cdef float ax = anisotropy[2]
  cdef float ay = anisotropy[1]
  cdef float az = anisotropy[0]

  if order == 'F':
    sx, sy, sz = sz, sy, sx
    ax = anisotropy[0]
    ay = anisotropy[1]
    az = anisotropy[2]

  data = data.astype(np.single).copy()
  cdef float[:, :, :] arr_memview = data

  _expand_3d_edt(
    <float*>&arr_memview[0,0,0],
    sx, sy, sz,
    ax, ay, az,
    parallel
  )

  return data
