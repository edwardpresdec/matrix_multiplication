#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>

#define MAX_SIZE 512

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
//M is a m x n matrix, N is a n x k matrix, then P is a m x k matrix
int main(){
    //initialize arrays
    float *M, *N, *P, *P_cpu;
    int M_rows = 512, M_cols = 512;
    int N_rows = 512, N_cols = 512;
    int P_rows = M_rows;
    int P_cols = N_cols;

    int M_size = M_rows * M_cols;
    int N_size = N_rows * N_cols;
    int P_size = P_rows * P_cols;

    // timing
    int K = 10;
    double runtime_array[K];
    double AVG = 0;
    clock_t start, stop;
    double runtime;

    for(int k = 0;k<K;k++){

        float* M = (float*)malloc(M_size * sizeof(float));
        float* N = (float*)malloc(N_size * sizeof(float));
        float* P = (float*)malloc(P_size * sizeof(float));
        float* P_cpu = (float*)malloc(P_size * sizeof(float));

        initializeMatrix(M, M_rows, M_cols);
        initializeMatrix(N, N_rows, N_cols);

        start = clock();

        #pragma acc kernels copyin(M[0:M_size],N[0:N_size]) copyout(P[0:P_size])
        {
            #pragma acc loop independent
            for (int i = 0; i < M_rows; i++) {
              #pragma acc loop independent
              for (int j = 0; j < N_cols; j++) {
                float sum = 0;
                #pragma acc loop independent reduction(+:sum)
                for (int k = 0; k < M_cols; k++) {

                    sum += M[i*M_rows +k] * N[k*N_cols + j];
                 }
                 P[i*P_rows+j] = sum;
              }

            }


        }

        stop = clock();
        runtime = ((double)(stop-start))/CLOCKS_PER_SEC;

        // Check results with sequential matrix multiplication
        matrixMultiplyCPU(M, N, P_cpu, M_rows, M_cols, N_cols);

        // Free memory
        free(M);
        free(N);
        free(P);
        free(P_cpu);

        printf("Parallel GPU runtime = %f ms\n", (runtime*1000)/4);
        AVG += runtime;
    }

    printf("Average Parallel runtime = %f ms\n",(AVG/K*1000)/4);
    return 0;
}
