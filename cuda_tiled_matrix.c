#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>

#define MAX_SIZE 512
#define TILE_WIDTH 32

//M is a m x n matrix, N is a n x k matrix, then P is a m x k matrix
__global__ void MMKernelTiled(float *d_M, float *d_N, float *d_P, int m, int k, int n) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Pvalue = 0.0;

    // Loop over the M and N tiles required to compute the P element
    for (int t = 0; t < (k - 1) / TILE_WIDTH + 1; ++t) {
        // Collaborative loading of M and N tiles into shared memory

        if (Row < m && t * TILE_WIDTH + tx < k)
            Mds[ty][tx] = d_M[Row * k + t * TILE_WIDTH + tx];
        else
            Mds[ty][tx] = 0.0;

        if (Col < n && t * TILE_WIDTH + ty < k)
            Nds[ty][tx] = d_N[(t * TILE_WIDTH + ty) * n + Col];
        else
            Nds[ty][tx] = 0.0;

        __syncthreads();  // Synchronize to ensure all tiles are loaded

        // Perform tile multiplication
        for (int i = 0; i < TILE_WIDTH; ++i)
            Pvalue += Mds[ty][i] * Nds[i][tx];

        __syncthreads();  // Synchronize to ensure all calculations are done before next load
    }

    // Write the block sub-matrix to global memory
    if (Row < m && Col < n)
        d_P[Row * n + Col] = Pvalue;
}

void initializeMatrix(float *arr, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        arr[i] = rand() / (float)RAND_MAX;
    }
}

bool check_equal(float* A1, float* A2, int rows, int cols){
  for(int i = 0; i < rows; i++)
    for(int j = 0; j < cols; j++){
      if(abs(A1[i * cols + j] - A2[i * cols + j]) > 0.001){
          return false;
      }
    }

  return true;
}

// Sequential matrix multiplication function
void matrixMultiplyCPU(float *A, float *B, float *C, int numARows, int numAColumns, int numBColumns) {
    for (int row = 0; row < numARows; row++) {
        for (int col = 0; col < numBColumns; col++) {
            float sum = 0.0;
            for (int i = 0; i < numAColumns; i++) {
                sum += A[row * numAColumns + i] * B[i * numBColumns + col];
            }
            C[row * numBColumns + col] = sum;
        }
    }
}

int main(){
    //initialize arrays
    float *M, *N, *P, *P_cpu;
    int M_rows = 512, M_cols = 256;
    int N_rows = 256, N_cols = 128;
    int P_rows = M_rows;
    int P_cols = N_cols;

    int M_size = M_rows * M_cols;
    int N_size = N_rows * N_cols;
    int P_size = P_rows * P_cols;

    int K = 10;
    double runtime_array[K];
    double AVG = 0;

    for(int k = 0;k<K;k++){
        // Allocate Unified Memory: accessible from CPU or GPU
        cudaMallocManaged(&M, M_size*sizeof(float));
        cudaMallocManaged(&N, N_size*sizeof(float));
        cudaMallocManaged(&P, P_size*sizeof(float));
        cudaMallocManaged(&P_cpu, P_size*sizeof(float));

        // timing
        cudaEvent_t start, stop;
        float runtime = 0;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        initializeMatrix(M, M_rows, M_cols);
        initializeMatrix(N, N_rows, N_cols);

        dim3 dimGrid(ceil(N_cols / float(TILE_WIDTH)), ceil(M_rows / float(TILE_WIDTH)), 1);
        dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

        cudaEventRecord(start, 0);
        MMKernelTiled<<<dimGrid, dimBlock>>>(M, N, P, M_rows, M_cols, N_cols);
        cudaEventRecord(stop, 0);

        cudaDeviceSynchronize();

        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&runtime, start, stop);

        // Check for kernel errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        }

        // Check results with sequential matrix multiplication
        matrixMultiplyCPU(M, N, P_cpu, M_rows, M_cols, N_cols);

        // Free memory
        cudaFree(M);
        cudaFree(N);
        cudaFree(P);
        cudaFree(P_cpu);

        printf("Parallel GPU runtime (With Tiling) = %f ms\n", runtime);
        if(k!=0){
            AVG += runtime;
        }
    }

    printf("Average Parallel runtime (WITH TILING) = %f ms\n",AVG/K);

    return 0;
}