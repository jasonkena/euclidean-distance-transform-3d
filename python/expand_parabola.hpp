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

template <typename T>
void _expand_1d_edt(T *d, const int n, const long int stride,
                    const float anisotropy) {
  // NOTE: d is already squared

  long int i;
  float w2 = anisotropy * anisotropy;

  for (i = stride; i < n * stride; i += stride) {
    d[i] = std::fmaxf(d[i - stride] - w2, d[i]);
  }

  for (i = (n - 2) * stride; i >= 0; i -= stride) {
    d[i] = std::fmaxf(d[i], d[i + stride] - w2);
  }
}

template <typename T, typename T2>
void _nearest_1d(T *d, T2 *v, const int n, const long int stride,
                 const float anisotropy) {
  // NOTE: d is already squared

  long int i;
  float w2 = anisotropy * anisotropy;

  T temp;

  for (i = stride; i < n * stride; i += stride) {
    temp = d[i - stride] - w2;

    if (temp > d[i]) {
      v[i] = v[i - stride];
    } else if (temp == d[i]) {
      v[i] = std::fmaxf(v[i], v[i - stride]);
    }

    d[i] = std::fmaxf(temp, d[i]);
  }

  for (i = (n - 2) * stride; i >= 0; i -= stride) {
    temp = d[i + stride] - w2;
    if (temp > d[i]) {
      v[i] = v[i + stride];
    } else if (temp == d[i]) {
      v[i] = std::fmaxf(v[i], v[i + stride]);
    }

    d[i] = std::fmaxf(d[i], temp);
  }
}

template <typename T>
void _expand_3d_edt(T *core_squared, const size_t sx, const size_t sy,
                    const size_t sz, const float wx, const float wy,
                    const float wz, const int parallel = 1) {

  const size_t sxy = sx * sy;

  size_t x, y, z;

  assert(core_squared != NULL);

  ThreadPool pool(parallel);

  size_t offset;
  for (z = 0; z < sz; z++) {
    for (y = 0; y < sy; y++) {
      offset = sx * y + sxy * z;
      pool.enqueue([core_squared, sx, wx, offset]() {
        _expand_1d_edt((core_squared + offset), sx, 1, wx);
      });
    }
  }

  pool.join();
  pool.start(parallel);

  for (z = 0; z < sz; z++) {
    for (x = 0; x < sx; x++) {
      offset = x + sxy * z;
      pool.enqueue([core_squared, sx, sy, wy, offset]() {
        _expand_1d_edt((core_squared + offset), sy, sx, wy);
      });
    }
  }

  pool.join();
  pool.start(parallel);

  for (y = 0; y < sy; y++) {
    for (x = 0; x < sx; x++) {
      offset = x + sx * y;
      pool.enqueue([core_squared, sz, sxy, wz, offset]() {
        _expand_1d_edt((core_squared + offset), sz, sxy, wz);
      });
    }
  }

  pool.join();
}

template <typename T, typename T2>
void _nearest_3d(T *centers, T2 *vals, const size_t sx, const size_t sy,
                 const size_t sz, const float wx, const float wy,
                 const float wz, const int parallel = 1) {

  const size_t sxy = sx * sy;
  const size_t voxels = sxy * sz;

  size_t x, y, z;

  assert(centers != NULL && vals != NULL);

  const float max_val =
      std::pow(sx * wx, 2) + std::pow(sy * wy, 2) + std::pow(sz * wz, 2);
  for (size_t i = 0; i < voxels; i++) {
    if (centers[i] > 0) {
      centers[i] = max_val;
    }
  }

  ThreadPool pool(parallel);

  size_t offset;
  for (z = 0; z < sz; z++) {
    for (y = 0; y < sy; y++) {
      offset = sx * y + sxy * z;
      pool.enqueue([centers, vals, sx, wx, offset]() {
        _nearest_1d((centers + offset), (vals + offset), sx, 1, wx);
      });
    }
  }

  pool.join();
  pool.start(parallel);

  for (z = 0; z < sz; z++) {
    for (x = 0; x < sx; x++) {
      offset = x + sxy * z;
      pool.enqueue([centers, vals, sx, sy, wy, offset]() {
        _nearest_1d((centers + offset), (vals + offset), sy, sx, wy);
      });
    }
  }

  pool.join();
  pool.start(parallel);

  for (y = 0; y < sy; y++) {
    for (x = 0; x < sx; x++) {
      offset = x + sx * y;
      pool.enqueue([centers, vals, sz, sxy, wz, offset]() {
        _nearest_1d((centers + offset), (vals + offset), sz, sxy, wz);
      });
    }
  }

  pool.join();
}
} // namespace py_expand_parabola
