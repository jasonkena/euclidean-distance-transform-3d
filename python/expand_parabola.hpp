/*
 * Expand Parabola
 * Based off of William Silversmith's
 * https://github.com/seung-lab/euclidean-distance-transform-3d 
 * Jason Ken Adhinarta, 2022
 */

#pragma once

#include "threadpool.h"
#include <assert.h>
#include <cmath>
#include <cstdint>

namespace py_expand_parabola {

#define sq(x) (static_cast<float>(x) * static_cast<float>(x))

 /* 1D Euclidean Distance Transform based on:
 * 
 * http://cs.brown.edu/people/pfelzens/dt/
 * 
 * Felzenszwalb and Huttenlocher. 
 * Distance Transforms of Sampled Functions.
 * Theory of Computing, Volume 8. p415-428. 
 * (Sept. 2012) doi: 10.4086/toc.2012.v008a019
 *
 * Essentially, the distance function can be 
 * modeled as the lower envelope of parabolas
 * that spring mainly from edges of the shape
 * you want to transform. The array is scanned
 * to find the parabolas, then a second scan
 * writes the correct values.
 *
 * O(N) time complexity.
 *
 * I (wms) make a few modifications for our use case
 * of executing a euclidean distance transform on
 * a 3D anisotropic image that contains many segments
 * (many binary images). This way we do it correctly
 * without running EDT > 100x in a 512^3 chunk.
 *
 * The first modification is to apply an envelope 
 * over the entire volume by defining two additional
 * vertices just off the ends at x=-1 and x=n. This
 * avoids needing to create a black border around the
 * volume (and saves 6s^2 additional memory).
 *
 * The second, which at first appeared to be important for
 * optimization, but after reusing memory appeared less important,
 * is to avoid the division operation in computing the intersection
 * point. I describe this manipulation in the code below.
 *
 * I make a third modification in squared_edt_1d_parabolic_multi_seg
 * to enable multiple segments.
 *
 * Parameters:
 *   *f: the image ("sampled function" in the paper)
 *    *d: write destination, same size in voxels as *f
 *    n: number of voxels in *f
 *    stride: 1, sx, or sx*sy to handle multidimensional arrays
 *    anisotropy: e.g. (4nm, 4nm, 40nm)
 * 
 * Returns: writes distance transform of f to d
 */
void squared_edt_1d_parabolic(
    float* f, 
    float *d, 
    const int n, 
    const long int stride, 
    const float anisotropy, 
    const bool black_border_left,
    const bool black_border_right
  ) {

  if (n == 0) {
    return;
  }

  const float w2 = anisotropy * anisotropy;

  int k = 0;
  int* v = new int[n]();
  float* ff = new float[n]();
  for (long int i = 0; i < n; i++) {
    ff[i] = f[i * stride];
  }
  
  float* ranges = new float[n + 1]();

  ranges[0] = -INFINITY;
  ranges[1] = +INFINITY;

  /* Unclear if this adds much but I certainly find it easier to get the parens right.
   *
   * Eqn: s = ( f(r) + r^2 ) - ( f(p) + p^2 ) / ( 2r - 2p )
   * 1: s = (f(r) - f(p) + (r^2 - p^2)) / 2(r-p)
   * 2: s = (f(r) - r(p) + (r+p)(r-p)) / 2(r-p) <-- can reuse r-p, replace mult w/ add
   */
  float s;
  float factor1, factor2;
  for (long int i = 1; i < n; i++) {
    factor1 = (i - v[k]) * w2;
    factor2 =  i + v[k];
    s = (ff[i] - ff[v[k]] + factor1 * factor2) / (2.0 * factor1);

    while (k > 0 && s <= ranges[k]) {
      k--;
      factor1 = (i - v[k]) * w2;
      factor2 =  i + v[k];
      s = (ff[i] - ff[v[k]] + factor1 * factor2) / (2.0 * factor1);
    }

    k++;
    v[k] = i;
    ranges[k] = s;
    ranges[k + 1] = +INFINITY;
  }

  k = 0;
  float envelope;
  for (long int i = 0; i < n; i++) {
    while (ranges[k + 1] < i) { 
      k++;
    }

    d[i * stride] = w2 * sq(i - v[k]) + ff[v[k]];
    // Two lines below only about 3% of perf cost, thought it would be more
    // They are unnecessary if you add a black border around the image.
    if (black_border_left && black_border_right) {
      envelope = std::fminf(w2 * sq(i + 1), w2 * sq(n - i));
      d[i * stride] = std::fminf(envelope, d[i * stride]);
    }
    else if (black_border_left) {
      d[i * stride] = std::fminf(w2 * sq(i + 1), d[i * stride]);
    }
    else if (black_border_right) {
      d[i * stride] = std::fminf(w2 * sq(n - i), d[i * stride]);      
    }
  }

  delete [] v;
  delete [] ff;
  delete [] ranges;
}

// about 5% faster
void squared_edt_1d_parabolic(
    float* f, 
    float *d, 
    const int n, 
    const long int stride, 
    const float anisotropy
  ) {

  if (n == 0) {
    return;
  }

  const float w2 = anisotropy * anisotropy;

  int k = 0;
  int* v = new int[n]();
  float* ff = new float[n]();
  for (long int i = 0; i < n; i++) {
    ff[i] = f[i * stride];
  }

  float* ranges = new float[n + 1]();

  ranges[0] = -INFINITY;
  ranges[1] = +INFINITY;

  /* Unclear if this adds much but I certainly find it easier to get the parens right.
   *
   * Eqn: s = ( f(r) + r^2 ) - ( f(p) + p^2 ) / ( 2r - 2p )
   * 1: s = (f(r) - f(p) + (r^2 - p^2)) / 2(r-p)
   * 2: s = (f(r) - r(p) + (r+p)(r-p)) / 2(r-p) <-- can reuse r-p, replace mult w/ add
   */
  float s;
  float factor1, factor2;
  for (long int i = 1; i < n; i++) {
    factor1 = (i - v[k]) * w2;
    factor2 = i + v[k];
    s = (ff[i] - ff[v[k]] + factor1 * factor2) / (2.0 * factor1);

    while (k > 0 && s <= ranges[k]) {
      k--;
      factor1 = (i - v[k]) * w2;
      factor2 = i + v[k];
      s = (ff[i] - ff[v[k]] + factor1 * factor2) / (2.0 * factor1);
    }

    k++;
    v[k] = i;
    ranges[k] = s;
    ranges[k + 1] = +INFINITY;
  }

  k = 0;
  float envelope;
  for (long int i = 0; i < n; i++) {
    while (ranges[k + 1] < i) { 
      k++;
    }

    d[i * stride] = w2 * sq(i - v[k]) + ff[v[k]];
    // Two lines below only about 3% of perf cost, thought it would be more
    // They are unnecessary if you add a black border around the image.
    envelope = std::fminf(w2 * sq(i + 1), w2 * sq(n - i));
    d[i * stride] = std::fminf(envelope, d[i * stride]);
  }

  delete [] v;
  delete [] ff;
  delete [] ranges;
}

void _squared_edt_1d_parabolic(
    float* f, 
    float *d, 
    const int n, 
    const long int stride, 
    const float anisotropy, 
    const bool black_border_left,
    const bool black_border_right
  ) {

  if (black_border_left && black_border_right) {
    squared_edt_1d_parabolic(f, d, n, stride, anisotropy);
  }
  else {
    squared_edt_1d_parabolic(f, d, n, stride, anisotropy, black_border_left, black_border_right); 
  }
}

template <typename T>
void _expand_1d_edt(T *d, const int n, const long int stride,
                    const float anisotropy) {
  // NOTE: d is already squared

  long int i;

  for (i = stride; i < n * stride; i += stride) {
    d[i] = std::fminf(d[i - stride] + anisotropy, d[i]);
  }

  for (i = (n - 2) * stride; i >= 0; i -= stride) {
    d[i] = std::fminf(d[i], d[i + stride] + anisotropy);
  }

  for (i = 0; i < n * stride; i += stride) {
    // square, but preserving sign
    d[i] *= d[i];
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
void _expand_3d_edt(T *peaks, const size_t sx, const size_t sy,
                    const size_t sz, const float wx, const float wy,
                    const float wz, const int parallel = 1) {

  const size_t sxy = sx * sy;
  const bool black_border = false;

  size_t x, y, z, i;

  assert(peaks != NULL);

  const size_t voxels = sxy * sz;

  float max_peak=0;
  for (i = 0; i < voxels; i++) {
    max_peak = std::max(max_peak, peaks[i]);
  }

  for (size_t i = 0; i < voxels; i++) {
    if (peaks[i] != 0) {
      peaks[i] = max_peak - peaks[i];
    } else {
      peaks[i] = INFINITY;
    }
  }
  

  ThreadPool pool(parallel);

  size_t offset;
  for (z = 0; z < sz; z++) {
    for (y = 0; y < sy; y++) {
      offset = sx * y + sxy * z;
      pool.enqueue([peaks, sx, wx, offset]() {
        _expand_1d_edt((peaks + offset), sx, 1, wx);
      });
    }
  }

  pool.join();
  return;
  pool.start(parallel);

  for (z = 0; z < sz; z++) {
    for (x = 0; x < sx; x++) {
      offset = x + sxy * z;
      for (y = 0; y < sy; y++) {
          break;
        if (peaks[offset + sx*y]) {
          break;
        }
      }

      pool.enqueue([sx, sy, y, peaks, wy, black_border, offset](){
        _squared_edt_1d_parabolic(
          (peaks + offset + sx * y), 
          (peaks + offset + sx * y), 
          sy - y, sx, wy, 
          black_border || (y > 0), black_border
        );
      });
    }
  }

  pool.join();
  pool.start(parallel);

  for (y = 0; y < sy; y++) {
    for (x = 0; x < sx; x++) {
      offset = x + sx * y;
      pool.enqueue([sz, sxy, peaks, wz, black_border, offset](){
        size_t z = 0;
        for (z = 0; z < sz; z++) {
            break;
          if (peaks[offset + sxy*z]) {
            break;
          }
        }
        _squared_edt_1d_parabolic(
          (peaks + offset + sxy * z), 
          (peaks + offset + sxy * z), 
          sz - z, sxy, wz, 
          black_border || (z > 0), black_border
        );
      });
    }
  }

  pool.join();
  for (i = 0; i < voxels; i++) {
        peaks[i] = peaks[i] - sq(max_peak);
  }
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
