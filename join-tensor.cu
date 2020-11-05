
#include <stdio.h>
#include <curand.h>
#include <cublas_v2.h>
#include <vector>
#include <math.h>


// Define some error checking macros.
#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
   if (stat != cudaSuccess) {
      fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
   }
}

#define cublasErrCheck(stat) { cublasErrCheck_((stat), __FILE__, __LINE__); }
void cublasErrCheck_(cublasStatus_t stat, const char *file, int line) {
   if (stat != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "cuBLAS Error: %d %s %d\n", stat, file, line);
   }
}

#define curandErrCheck(stat) { curandErrCheck_((stat), __FILE__, __LINE__); }
void curandErrCheck_(curandStatus_t stat, const char *file, int line) {
   if (stat != CURAND_STATUS_SUCCESS) {
      fprintf(stderr, "cuRand Error: %d %s %d\n", stat, file, line);
   }
}


__global__ void convertFp32ToFp16 (half *out, float *in, int n) {
   int idx = blockDim.x * blockIdx.x + threadIdx.x;
   if (idx < n) {
      out[idx] = in[idx];
   }
}

__global__ void condition_equal(float *in, float *out)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  out[index] = (((int)ceil(abs(((int)((in[index] + 0.001) * 100.0)/100.0) -1)))&&1) ^ 1;

}

int main(int argc, char* argv[]) {
    int MATRIX_M, MATRIX_N, MATRIX_K;
    int size1= atoi(argv[1]); 
  
    MATRIX_M = size1;
    MATRIX_N = size1;
    MATRIX_K = size1;

   float *c_host_cublas;
   cublasHandle_t cublasHandle;
   
   cudaEvent_t startcublas;
   cudaEvent_t stopcublas;

   cudaErrCheck(cudaEventCreate(&startcublas));
   cudaErrCheck(cudaEventCreate(&stopcublas));
   
   cublasErrCheck(cublasCreate(&cublasHandle));
   
   // Use tensor cores
   cublasErrCheck(cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH));

   //malloc - void *malloc(size_t size) allocates the requested memory and returns a pointer to it.c function
   c_host_cublas = (float*)malloc(MATRIX_M * MATRIX_N * sizeof(float));

   //create arrays with values
   float *h_a = new float[MATRIX_M*MATRIX_N];
   float *h_b = new float[MATRIX_M*MATRIX_N];
   float *h_c = new float[MATRIX_M*MATRIX_N];

   //TODO - create a array with initial values
    for(int i=0; i<MATRIX_M*MATRIX_N;i++){
        h_a[i] = 0;
        h_b[i] = 0;
        h_c[i] = 0;
    }
    float x =1;
    for(int i=MATRIX_M-1; i<MATRIX_M*MATRIX_N;i+=MATRIX_M){
       h_a[i] = (float)(1/(1/(x++)));
    }
    x=1;
    for(int i=MATRIX_M*MATRIX_N-MATRIX_M ;i<MATRIX_M*MATRIX_N;i++) {
        h_b[i] = (float)(1/(x++));
    }

    //printf("\n");
    //printf("Starting join operation, SELECT * FROM table1 NATURAL JOIN table2;");

	
    // printf("\n");
    // printf("\n------------Printing table A------------\n");
    // for(int i=0; i<MATRIX_M*MATRIX_N;i++){
    //     printf("%f ", h_a[i]);
    //     if((i+1)%MATRIX_M == 0){
    //     printf("\n");
    //     }
    // }

    // printf("\n------------Printing table B------------\n");
    // for(int i=0; i<MATRIX_M*MATRIX_N;i++){
    //     printf("%f ", h_b[i]);
    //     if((i+1)%MATRIX_M == 0){
    //     printf("\n");
    //     }
    // }
	

    //allocate memory and copy the values from CPU to GPU
   float *d_a, *d_b, *d_c;
   half *d_a_half, *d_b_half;
   half *h_a_half = new half[MATRIX_M*MATRIX_N];
   half *h_b_half = new half[MATRIX_M*MATRIX_N];


   cudaErrCheck(cudaMalloc((void**)&d_a, MATRIX_M * MATRIX_K * sizeof(float)));
   cudaErrCheck(cudaMalloc((void**)&d_b, MATRIX_K * MATRIX_N * sizeof(float)));
   cudaErrCheck(cudaMalloc((void**)&d_c, MATRIX_K * MATRIX_N * sizeof(float)));
   cudaErrCheck(cudaMalloc((void**)&d_a_half, MATRIX_K * MATRIX_N * sizeof(half)));
   cudaErrCheck(cudaMalloc((void**)&d_b_half, MATRIX_K * MATRIX_N * sizeof(half)));

   cudaErrCheck(cudaMemcpy(d_a, h_a, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyHostToDevice));
   cudaErrCheck(cudaMemcpy(d_b, h_b, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyHostToDevice));


   cudaErrCheck(cudaMemcpy(d_a_half, h_a_half, MATRIX_M * MATRIX_N * sizeof(half), cudaMemcpyHostToDevice));
   cudaErrCheck(cudaMemcpy(d_b_half, h_b_half, MATRIX_M * MATRIX_N * sizeof(half), cudaMemcpyHostToDevice));
   cudaErrCheck(cudaMemcpy(d_c, h_c, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyHostToDevice));
   
  
   convertFp32ToFp16 <<< (MATRIX_M * MATRIX_K + 255) / 256, 256 >>> (d_a_half, d_a, MATRIX_M*MATRIX_N);
   convertFp32ToFp16 <<< (MATRIX_M * MATRIX_K + 255) / 256, 256  >>> (d_b_half, d_b, MATRIX_M*MATRIX_N);
   
   float alpha = 1.0f;
   float beta = 0.0f;

  // printf("\nM = %d, N = %d, K = %d. alpha = %f, beta = %f\n\n", MATRIX_M, MATRIX_N, MATRIX_K, alpha, beta);

     // Now using cuBLAS
   //printf("------------Running with cuBLAS------------n");

   float cublasTime1;
   float cublasTime2;

   cudaErrCheck(cudaEventRecord(startcublas));
   cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
                MATRIX_M, MATRIX_N, MATRIX_K, 
                &alpha,
                d_b_half, CUDA_R_16F, MATRIX_M,
                d_a_half, CUDA_R_16F, MATRIX_K,
                &beta, 
                d_c, CUDA_R_32F, MATRIX_M,
                CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

   	cudaErrCheck(cudaEventRecord(stopcublas));
    cudaErrCheck(cudaEventSynchronize(stopcublas));
    cudaErrCheck(cudaEventElapsedTime(&cublasTime1, startcublas, stopcublas));
    //printf("Matrix multiplication took %fms\n", cublasTime1);
	  
   //Checking Results
   //printf("\n------------Checking results------------\n");
    //   cudaErrCheck(cudaMemcpy(c_host_cublas, d_c, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost));
    
    //    for (int i = 0; i < MATRIX_M*MATRIX_N; i++) {
    //        printf("%f ", c_host_cublas[i]);
    //        if((i+1)%MATRIX_M== 0){
    //          printf("\n");
    //        }
    //    }


	float *d_g;
	float size = MATRIX_M*MATRIX_N * sizeof( float );
	float  *m = (float *)malloc( size );

	cudaErrCheck(cudaMalloc( (void **) &d_g, size ));
	cudaErrCheck(cudaMemcpy( d_g, m, size, cudaMemcpyHostToDevice ));



	/* launch the kernel on the GPU */
	/* insert the launch parameters to launch the kernel properly using blocks and threads */ 

	cudaErrCheck(cudaEventRecord(startcublas));
	condition_equal<<< (MATRIX_M * MATRIX_K + 255) / 256, 256>>>(d_c,d_g);
	cudaErrCheck(cudaEventRecord(stopcublas));
	
    cudaErrCheck(cudaEventSynchronize(stopcublas));
    cudaErrCheck(cudaEventElapsedTime(&cublasTime2, startcublas, stopcublas));
    //printf("Bit flipping took %fms\n", cublasTime2);
	printf("%fms", cublasTime1 + cublasTime2);


	cudaErrCheck(cudaDeviceSynchronize());

	/* copy result back to host */
	/* fix the parameters needed to copy data back to the host */
	cudaErrCheck(cudaMemcpy(m, d_g, size, cudaMemcpyDeviceToHost ));
	





    //printf("\n------------Result of JOIN operation -- Printing top 15 results------------\n");
	// for (int i = 0; i < MATRIX_M*MATRIX_N; i++) {
    //    int v2 = m[i];
    //    printf("%d ", v2);
    //    //if(v2==1)
    //      //printf("Merge Row %d of Table 1 and Row %d of Table 2 \n",(int)(i/4), i%4 );
    //     //printf("%f %f %f %f\n", v2*original_table[i] , v2*original_table[i+4], v2*original_table[i+8], v2*original_table[i+12]);
    //    }

	/* clean up */

	free(m);
	cudaFree( d_g );

   cudaErrCheck(cudaEventDestroy(startcublas));             
   cudaErrCheck(cudaEventDestroy(stopcublas));
   
   //free the data
   cudaErrCheck(cudaFree(d_a));
   cudaErrCheck(cudaFree(d_b));
   cudaErrCheck(cudaFree(d_c));
   

   cudaErrCheck(cudaDeviceReset());
   return 0;
}