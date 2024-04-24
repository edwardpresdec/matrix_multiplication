#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>

#define MAX_SIZE 512
#define TILE_WIDTH 32

//M is a m x n matrix, N is a n x k matrix, then P is a m x k matrix
__global__ void MMKernel(float* M, float* N, float* P, int M_rows, int M_cols, int N_cols) { //M_cols == N_rows

    // Calculate the row index of the P element and M
    int Row = blockIdx.y*blockDim.y+threadIdx.y;

    // Calculate the column index of P and N
    int Col = blockIdx.x*blockDim.x+threadIdx.x;

    if ((Row < M_rows) && (Col < N_cols)) {
      float Pvalue = 0;
      // each thread computes one element of the block sub-matrix

      for (int k = 0; k < M_cols; ++k) {
          Pvalue += M[Row*M_cols+k]*N[k*N_cols+Col];
      }

      P[Row*N_cols+Col] = Pvalue;
    }
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
        MMKernel<<<dimGrid, dimBlock>>>(M, N, P, M_rows, M_cols, N_cols);
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

        printf("Parallel GPU runtime = %f ms\n", runtime);
        AVG += runtime;
    }

    printf("Average Parallel runtime = %f ms\n",AVG/K);
    return 0;
}