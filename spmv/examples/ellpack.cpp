/******************************************************************************
Copyright (c) 2023 Advanced Micro Devices, Inc. (AMD)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
******************************************************************************/
#include "hip/hip_runtime.h"
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/fill.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

/*
 * Run via:
 *  ./ellpack /PATH/TO/MATRIX.MTX NUM_TRIALS
 */
using namespace std;

#define HIP_CALL(call)                                     \
  do {                                                     \
    hipError_t err = call;                                 \
    if (hipSuccess != err) {                               \
      printf("HIP ERROR (code = %d, %s) at %s:%d\n", err,  \
             hipGetErrorString(err), __FILE__, __LINE__);  \
      assert(0);                                           \
      exit(1);                                             \
    }                                                      \
  } while (0)

/********************************************************
 * Some Helpful Function Prototypes
 ********************************************************/
unsigned long prevPowerOf2(unsigned long v);

vector<double> host_spmv(const int m, const int n, const int nnz,
                         const vector<int> & rows,
                         const vector<int> & cols,
                         const vector<double> & vals,
                         const vector<double> & x);

double norm2(const vector<double> &x);

void read_mm(const string filename, int& m, int& n, int& nnz, int& max_nnz_per_row,
             vector<int> & rows, vector<int> & cols, vector<double> & vals);

/********************************************************
 * CSR Ellpack Conversion Kernels
 ********************************************************/
template <int THREADS_PER_ROW>
__global__ void
fillColEll(const int m, const int max_nnz_per_row, const int *__restrict__ rows,
           const int *__restrict__ cols, const double *__restrict__ vals,
           int *ell_cols, double *ell_vals) {
  const int lane = threadIdx.x % THREADS_PER_ROW;
  const int row = threadIdx.x / THREADS_PER_ROW +
                  (blockDim.x / THREADS_PER_ROW) * blockIdx.x;

  if (row < m) {
    /* determine the start and ends of each row */
    int r[2];
    if (lane < 2)
      r[lane] = rows[row + lane];
    r[0] = __shfl(r[0], 0, THREADS_PER_ROW);
    r[1] = __shfl(r[1], 1, THREADS_PER_ROW);
    int maxcol = -1;
    for (int i = r[0] + lane; i < r[1]; i += THREADS_PER_ROW) {
      int col = cols[i];
      double val = vals[i];
      ell_cols[(i - r[0]) * m + row] = col;
      ell_vals[(i - r[0]) * m + row] = val;
      if (col > maxcol)
        maxcol = col;
    }
    for (int i = THREADS_PER_ROW >> 1; i > 0; i >>= 1)
      maxcol = max(maxcol, __shfl_down(maxcol, i, THREADS_PER_ROW));
    maxcol = __shfl(maxcol, 0, THREADS_PER_ROW);
    for (int i = r[1] - r[0] + lane; i < max_nnz_per_row;
         i += THREADS_PER_ROW) {
      ell_cols[i * m + row] = maxcol;
      ell_vals[i * m + row] = 0.0;
    }
  }
}

/********************************************************
 * Ellpack SpMV Kernel
 ********************************************************/
__global__ void ellpack_kernel(const int m, const int max_nnz_per_row,
                               const int *__restrict__ cols,
                               const double *__restrict__ vals,
                               const double *__restrict__ x, double *__restrict__ y,
                               const double alpha, const double beta) {
  const int row = threadIdx.x + blockDim.x * blockIdx.x;

  if (row < m) {
    double sum = 0;
    for (int i = row; i < max_nnz_per_row * m; i += m) {
      sum += vals[i] * x[cols[i]];
    }

    /* write to memory */
    if (beta == 0) {
      y[row] = alpha * sum;
    } else {
      y[row] = alpha * sum + beta * y[row];
    }
  }
  return;
}

/********************************************************
 * main executable
 ********************************************************/
int main(int argc, char const *argv[]) {

  if (argc < 3) {
    cout << "Usage: ./ellpack <matrix market file> <num trials>" << endl;
    return 1;
  }
  string matfile = string(argv[1]);
  int numTrials = atoi(argv[2]);

  /* Device Properties */
  int deviceId = 0;
  HIP_CALL(hipGetDevice(&deviceId));

  hipDeviceProp_t prop;
  HIP_CALL(hipGetDeviceProperties(&prop, deviceId));
  int warpSize = prop.warpSize;

  /* get matrix */
  int m, n, nnz, max_nnz_per_row;
  vector<int> hrows(0);
  vector<int> hcols(0);
  vector<double> hvals(0);
  read_mm(matfile, m, n, nnz, max_nnz_per_row, hrows, hcols, hvals);

  /* create randomized, host lhs vector, hx */
  vector<double> hx(n);
  vector<double> hy(m);
  srand(static_cast<unsigned int>(time(NULL)));
  for (int i = 0; i < n; ++i) hx[i] = (rand() * 1.0) / RAND_MAX;

  /* Allocate device space and copy to */
  int *drows = NULL;
  int *dcols = NULL;
  double *dvals = NULL;
  HIP_CALL(hipMalloc((void **)&drows, sizeof(int) * (m + 1)));
  HIP_CALL(hipMalloc((void **)&dcols, sizeof(int) * nnz));
  HIP_CALL(hipMalloc((void **)&dvals, sizeof(double) * nnz));

  HIP_CALL(hipMemcpy(drows, hrows.data(), sizeof(int) * (m + 1),
               hipMemcpyHostToDevice));
  HIP_CALL(hipMemcpy(dcols, hcols.data(), sizeof(int) * nnz,
               hipMemcpyHostToDevice));
  HIP_CALL(hipMemcpy(dvals, hvals.data(), sizeof(double) * nnz,
               hipMemcpyHostToDevice));
  double *dx = NULL;
  double *dy = NULL;
  HIP_CALL(hipMalloc((void **)&dx, sizeof(double) * n));
  HIP_CALL(hipMalloc((void **)&dy, sizeof(double) * m));
  HIP_CALL(hipMemcpy(dx, hx.data(), sizeof(double) * n, hipMemcpyHostToDevice));

  /* Determine Launch Parameters for format conversion */
  int threads_per_block=256;
  int nnz_per_row = nnz / m;
  int threads_per_row = prevPowerOf2(nnz_per_row);
  int rows_per_block = threads_per_block >= threads_per_row
    ? threads_per_block / threads_per_row : 1;
  int num_blocks = (m + rows_per_block - 1) / rows_per_block;

  /* Convert to Ellpack */
  HIP_CALL(hipDeviceSynchronize());
  auto start = std::chrono::high_resolution_clock::now();
  int *dellcols = NULL;
  double *dellvals = NULL;
  HIP_CALL(hipMalloc((void **)&dellcols, sizeof(int) * max_nnz_per_row * m));
  HIP_CALL(hipMalloc((void **)&dellvals, sizeof(double) * max_nnz_per_row * m));
  HIP_CALL(hipMemset(dellcols, 0, sizeof(int) * max_nnz_per_row * m));
  HIP_CALL(hipMemset(dellvals, 0, sizeof(double) * max_nnz_per_row * m));
  dim3 grid(num_blocks, 1, 1);
  dim3 block(threads_per_block, 1, 1);

  if (threads_per_row <= 2) {
    fillColEll<2><<<grid,block>>>(m, max_nnz_per_row, drows, dcols, dvals, dellcols, dellvals);
  } else if (threads_per_row <= 4) {
    fillColEll<4><<<grid,block>>>(m, max_nnz_per_row, drows, dcols, dvals, dellcols, dellvals);
  } else if (threads_per_row <= 8) {
    fillColEll<8><<<grid,block>>>(m, max_nnz_per_row, drows, dcols, dvals, dellcols, dellvals);
  } else if (threads_per_row <= 16) {
    fillColEll<16><<<grid,block>>>(m, max_nnz_per_row, drows, dcols, dvals, dellcols, dellvals);
  } else if (threads_per_row <= 32) {
    fillColEll<32><<<grid,block>>>(m, max_nnz_per_row, drows, dcols, dvals, dellcols, dellvals);
  } else {
    fillColEll<64><<<grid,block>>>(m, max_nnz_per_row, drows, dcols, dvals, dellcols, dellvals);
  }
  HIP_CALL(hipDeviceSynchronize());
  auto stop = std::chrono::high_resolution_clock::now();
  auto setupTime = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count() * 1.e-6;
  cout << "Ellpack Conversion time=" << setupTime << " ms" << endl;

  /* Determine Launch Parameters for SpMV */
  double alpha = 1.0, beta = 0.0;
  num_blocks = (m + threads_per_block - 1) / threads_per_block;

  /* Call Ellpack SpMV */
  auto avg_timing = 0.0;
  auto avg_bandwidth = 0.0;
  for (int i=0; i<numTrials; ++i)
  {
    auto start = chrono::high_resolution_clock::now();
    ellpack_kernel<<<grid,block>>>(m, max_nnz_per_row, dellcols, dellvals, dx, dy, alpha, beta);
    HIP_CALL(hipDeviceSynchronize());
    auto stop = chrono::high_resolution_clock::now();

    auto timing = chrono::duration_cast<chrono::nanoseconds>(stop - start).count() * 1.e-6;
    uint64_t mem_size = (sizeof(double) + sizeof(int)) * nnz + sizeof(int) * (m + 1) + sizeof(double) * (n + m);
    auto bandwidth = mem_size / timing * 1.e-9;
    cout << "Trial " << i << " of " << numTrials << " Ellpack SpMV time=" << timing << " ms, Bandwidth=" << bandwidth << " TB/s\n";
    avg_timing += timing;
    avg_bandwidth += bandwidth;
  }
  cout << "Average Ellpack SpMV time=" << avg_timing / numTrials << " ms, Average Bandwidth=" << avg_bandwidth / numTrials << " TB/s\n";

  /* copy back to host */
  HIP_CALL(hipMemcpy(hy.data(), dy, sizeof(double) * m, hipMemcpyDeviceToHost));
  cout << "||gpu spmv result||_2 = " << norm2(hy) << endl;

  vector<double> baseline_y = host_spmv(m,n,nnz,hrows,hcols,hvals,hx);
  cout << "||host_spmv result||_2 = " << norm2(baseline_y) << endl;

  /* Clear up on device */
  HIP_CALL(hipFree(dellcols));
  HIP_CALL(hipFree(dellvals));
  HIP_CALL(hipFree(drows));
  HIP_CALL(hipFree(dcols));
  HIP_CALL(hipFree(dvals));
  HIP_CALL(hipFree(dx));
  HIP_CALL(hipFree(dy));
  return 0;
}


void read_mm(const string filename, int& m, int& n, int& nnz, int& max_nnz_per_row,
             vector<int> & rows, vector<int> & cols, vector<double> & vals)
{
  ifstream file(filename);
  int ind = filename.rfind("/");
  string matname = filename.substr(ind+1);

  /* Ignore comments headers */
  char line[2048];
  bool foundSymm = false;
  while (file.peek() == '%') {
    file.getline(line, 2048);
    string s1(line);
    if (s1.find(" symmetric") != string::npos)
      foundSymm = true;
  }
  file >> m >> n >> nnz;
  int nnzSymm = nnz;
  if (foundSymm) {
    cout << "Matrix, " << matname << ", is symmetric" << endl;
    cout << "shape=" << m << " " << n << " " << nnz << " " << 2 * nnz - m << endl;
    nnz = 2 * nnz - m;
  } else {
    cout << "Matrix, " << matname << ", is general" << endl;
    cout << "shape=" << m << " " << n << " " << nnz << endl;
  }
  /* Read number of rows and columns */
  rows.resize(m+1);
  cols.resize(nnz);
  vals.resize(nnz);
  vector<int> coo_rows(nnz);

  /* fill the matrix with data */
  if (foundSymm) {
    int k = 0;
    for (int l = 0; l < nnzSymm; ++l) {
      int rowp1;
      int colp1;
      double value;
      /* MM format uses 1 based indexing */
      file >> rowp1 >> colp1 >> value;
      if (colp1 != rowp1) {
        coo_rows[k] = rowp1 - 1;
        cols[k] = colp1 - 1;
        vals[k] = value;
        k++;
        cols[k] = rowp1 - 1;
        coo_rows[k] = colp1 - 1;
        vals[k] = value;
        k++;
      } else {
        coo_rows[k] = colp1 - 1;
        cols[k] = rowp1 - 1;
        vals[k] = value;
        k++;
      }
    };
  } else {
    for (int l = 0; l < nnz; ++l) {
      int rowp1;
      int colp1;
      /* MM format uses 1 based indexing */
      file >> rowp1 >> colp1 >> vals[l];
      coo_rows[l] = rowp1 - 1;
      cols[l] = colp1 - 1;
    };
  }
  file.close();

  /* sort the coo matrix */
  auto begin_keys =
    thrust::make_zip_iterator(thrust::make_tuple(coo_rows.data(),       cols.data()));
  auto end_keys =
    thrust::make_zip_iterator(thrust::make_tuple(coo_rows.data() + nnz, cols.data() + nnz));

  thrust::stable_sort_by_key(thrust::host, begin_keys, end_keys, vals.data(),
                    thrust::less<thrust::tuple<int, int>>());

  /* fill the row counts */
  thrust::fill(thrust::host, rows.data(), rows.data()+m+1, 0);
  for (int nz = 0; nz < nnz; ++nz)
    rows[coo_rows[nz]]++;

  /* calculate max_nnz_per_row */
  max_nnz_per_row=0;
  for (int i = 0; i < m; ++i)
    if (rows[i]>max_nnz_per_row)
      max_nnz_per_row = rows[i];

  /* transform the row counts to row offsets */
  thrust::exclusive_scan(thrust::host, rows.data(), rows.data() + m + 1, rows.data());
}

unsigned long prevPowerOf2(unsigned long v) {
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;
  return v >> 1;
}

vector<double> host_spmv(const int m, const int n, const int nnz,
                         const vector<int> & rows,
                         const vector<int> & cols,
                         const vector<double> & vals,
                         const vector<double> & x) {
  vector<double> y(m);
  for (int row = 0; row < m; row++) {
    y[row] = 0.;
    for (int nz = rows[row]; nz < rows[row + 1]; ++nz) {
      y[row] += vals[nz] * x[cols[nz]];
    }
  }
  return y;
}

double norm2(const vector<double> &x) {
  double sum = 0.;
  for (size_t i = 0; i < x.size(); ++i)
    sum += x[i] * x[i];
  return sqrt(sum);
}
