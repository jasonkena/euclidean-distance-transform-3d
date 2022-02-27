/*
 * Expand Parabola
 * Based off of William Silversmith's
 * https://github.com/seung-lab/euclidean-distance-transform-3d Jason Ken
 * Adhinarta, 2022
 */

#pragma once

#include "threadpool.h"
#include <assert.h>
#include <cmath>
#include <cstdint>

namespace py_expand_parabola {

/* d* segids: float with values indicating height of parabola
 * n: size of segids, d
 * stride: typically 1, but can be used on a
 *    multi dimensional array, in which case it is nx, nx*ny, etc
 * anisotropy: physical distance of each voxel
 */
template <typename T>
void _expand_1d(T *d, const int n, const long int stride,
                const float anistropy) {
  // NOTE: d is already squared

  long int i;
  float w2 = anistropy * anistropy;

  for (i = stride; i < n * stride; i += stride) {
    d[i] = std::fmaxf(d[i - stride] - w2, d[i]);
  }

  for (i = (n - 2) * stride; i >= 0; i -= stride) {
    d[i] = std::fmaxf(d[i], d[i + stride] - w2);
  }
}

template <typename T>
void _expand_3d(T *core_squared, const size_t sx, const size_t sy,
                const size_t sz, const float wx, const float wy, const float wz,
                const int parallel = 1) {

  const size_t sxy = sx * sy;

  size_t x, y, z;

  assert(core_squared != NULL);

  ThreadPool pool(parallel);

  size_t offset;
  for (z = 0; z < sz; z++) {
    for (y = 0; y < sy; y++) {
      offset = sx * y + sxy * z;
      pool.enqueue([core_squared, sx, wx, offset]() {
        _expand_1d((core_squared + offset), sx, 1, wx);
      });
    }
  }

  pool.join();
  pool.start(parallel);

  for (z = 0; z < sz; z++) {
    for (x = 0; x < sx; x++) {
      offset = x + sxy * z;
      pool.enqueue([core_squared, sx, sy, wy, offset]() {
        _expand_1d((core_squared + offset), sy, sx, wy);
      });
    }
  }

  pool.join();
  pool.start(parallel);

  for (y = 0; y < sy; y++) {
    for (x = 0; x < sx; x++) {
      offset = x + sx * y;
      pool.enqueue([core_squared, sz, sxy, wz, offset]() {
        _expand_1d((core_squared + offset), sz, sxy, wz);
      });
    }
  }

  pool.join();
}
} // namespace py_expand_parabola
