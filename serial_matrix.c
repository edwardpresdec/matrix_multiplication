#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MAX_SIZE 512

// Function to multiply two matrices M and N, result is P
void matrixMultiply(int M[MAX_SIZE][MAX_SIZE], int N[MAX_SIZE][MAX_SIZE], int P[MAX_SIZE][MAX_SIZE], int M_rows, int M_cols, int N_rows, int N_cols) {
    if (M_cols != N_rows) {
        printf("Matrix multiplication not possible!\n");
        return;
    }

    for (int i = 0; i < M_rows; i++) {
        for (int j = 0; j < N_cols; j++) {
            P[i][j] = 0; // Initialize the element to zero
            for (int k = 0; k < M_cols; k++) {
                P[i][j] += M[i][k] * N[k][j];
            }
        }
    }
}

int main() {
    const int M_rows = 512, M_cols = 256;
    const int N_rows = 256, N_cols = 128;
    int M[MAX_SIZE][MAX_SIZE];
    int N[MAX_SIZE][MAX_SIZE];
    int P[MAX_SIZE][MAX_SIZE];

    // Seed the random number generator
    srand(time(NULL));

    // Initialize matrices M and N with random numbers
    printf("Matrix M:\n");
    for (int i = 0; i < M_rows; i++) {
        for (int j = 0; j < M_cols; j++) {
            M[i][j] = rand() % 100;  // Random number between 0 and 99
            //printf("%d ", M[i][j]);
        }
        //printf("\n");
    }

    printf("Matrix N:\n");
    for (int i = 0; i < N_rows; i++) {
        for (int j = 0; j < N_cols; j++) {
            N[i][j] = rand() % 100;  // Random number between 0 and 99
            //printf("%d ", N[i][j]);
        }
        //printf("\n");
    }
    int K = 10;
    double runtime_array[K];
    double AVG = 0;
    for(int k = 0;k<K;k++){
        clock_t start_time = clock();

        matrixMultiply(M, N, P, M_rows, M_cols, N_rows, N_cols);

        clock_t end_time = clock();
        double runtime = (double)(end_time - start_time) / CLOCKS_PER_SEC;
        runtime_array[k] = runtime;
        AVG += runtime;
    }



    // Display the resulting matrix P
    printf("Resultant matrix P after multiplication:\n");
    for (int i = 0; i < M_rows; i++) {
        for (int j = 0; j < N_cols; j++) {
            //printf("%d ", P[i][j]);
        }
        //printf("\n");
    }


    printf("Average sequential runtime: %fms",AVG/K *1000);

    return 0;
}

