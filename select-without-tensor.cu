


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


#define MATRIX_M 512
#define MATRIX_N 512
#define MATRIX_K 512


__global__ void convertFp32ToFp16 (half *out, float *in, int n) {
   int idx = blockDim.x * blockIdx.x + threadIdx.x;
   if (idx < n) {
      out[idx] = in[idx];
   }
}

__global__ void condition_equal(float *in, float *out)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
  out[index] = (((int)ceil(abs(in[index]-1)))&&1) ^ 1;
}

__global__ void condition_greaterthan(float *in, float *out)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
  out[index] = ((int)(ceil(in[index]-1)+ abs(ceil(in[index]-1)))) && 1;
}

__global__ void condition_lessthan(float *in, float *out)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
  out[index] = ((int)(floor(in[index]-1) - abs(floor(in[index]-1)))) && 1;
}

int main(int argc, char* argv[]) {

   float *c_host_cublas;
   int condition;
   cublasHandle_t cublasHandle;
   
   cudaEvent_t startcublas;
   cudaEvent_t stopcublas;

   cudaErrCheck(cudaEventCreate(&startcublas));
   cudaErrCheck(cudaEventCreate(&stopcublas));
   
   cublasErrCheck(cublasCreate(&cublasHandle));
   
   // Use tensor cores
   //cublasErrCheck(cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH));

   //malloc - void *malloc(size_t size) allocates the requested memory and returns a pointer to it.c function
   c_host_cublas = (float*)malloc(MATRIX_M * MATRIX_N * sizeof(float));

    printf("\n");
    printf("\n------------Original table ------------\n");
    
   float *original_table = new float[MATRIX_M*MATRIX_N];
   
   for(int i=0; i<MATRIX_M*MATRIX_N;i++){
            original_table[i] = i;
            //if(i%MATRIX_M==0)
                //printf("Column %d printing ---->", (i/MATRIX_M)+1);
            //printf(" %d", i);
            //if((i+1)%MATRIX_M == 0){ 
            //printf("\n");
            //}
            
    }

   
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
	    
    for(int i=0;i<MATRIX_M;i++) {
        h_a[i] = original_table[i];
        h_b[i] = 0.5;
    }

    printf("\n");
    printf("Starting select operation, SELECT * FROM table WHERE A ? 2 ");
    printf("\n\n Enter 1 for equality condition\n 2 for less than condition\n 3 for greater than condition\n");
    scanf("%d", &condition);
    printf("\n");
    //printf("\n------------Printing table A------------\n");
    /*for(int i=0; i<MATRIX_M*MATRIX_N;i++){
        //printf("%f ", h_a[i]);
        if((i+1)%MATRIX_M == 0){
        printf("\n");
        }
    }

    printf("\n------------Printing table B------------\n");
    for(int i=0; i<MATRIX_M*MATRIX_N;i++){
        //printf("%f ", h_b[i]);
        if((i+1)%MATRIX_N == 0){
        printf("\n");
        }
    }*/


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
   float beta = 1.0f;

   printf("\nM = %d, N = %d, K = %d. alpha = %f, beta = %f\n\n", MATRIX_M, MATRIX_N, MATRIX_K, alpha, beta);

   float cublasTime1;
   float cublasTime2;

     // Now using cuBLAS
   printf("------------Running with cuBLAS------------n");
   cudaErrCheck(cudaEventRecord(startcublas));
   cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
                MATRIX_M, MATRIX_N, MATRIX_K, 
                &alpha,
                d_a_half, CUDA_R_16F, MATRIX_M,
                d_b_half, CUDA_R_16F, MATRIX_K,
                &beta, 
                d_c, CUDA_R_32F, MATRIX_M,
                CUDA_R_32F,  CUBLAS_GEMM_DEFAULT));
   	cudaErrCheck(cudaEventRecord(stopcublas));
    cudaErrCheck(cudaEventSynchronize(stopcublas));
    cudaErrCheck(cudaEventElapsedTime(&cublasTime1, startcublas, stopcublas));
    printf("Matrix multiplication took %fms\n", cublasTime1);

   // Checking Results
   printf("\n------------Checking results------------\n");
   cudaErrCheck(cudaMemcpy(c_host_cublas, d_c, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost));
   
  //  for (int i = 0; i < MATRIX_M ; i++) {
  //      printf("%f ", c_host_cublas[i]);
  //      if((i+1)%MATRIX_M== 0){
  //        printf("\n");
  //      }
  //  }

    //----------------------------------------------------

	float *d_g;
	float size = MATRIX_M*MATRIX_N * sizeof( float );

	cudaErrCheck(cudaMalloc( (void **) &d_g, size ));
	cudaErrCheck(cudaMemcpy( d_g, h_c, size, cudaMemcpyHostToDevice ));

    cudaErrCheck(cudaEventRecord(startcublas));

	/* launch the kernel on the GPU */
	/* insert the launch parameters to launch the kernel properly using blocks and threads */ 
	  if(condition == 1)
		  condition_equal<<< 1, MATRIX_M >>>(d_c,d_g);
	  else if(condition == 2)
		condition_lessthan<<< 1, MATRIX_M>>>(d_c,d_g);
	  else
		 condition_greaterthan<<< 1, MATRIX_M>>>(d_c,d_g);

    cudaErrCheck(cudaEventRecord(stopcublas));
    cudaErrCheck(cudaEventSynchronize(stopcublas));
    cudaErrCheck(cudaEventElapsedTime(&cublasTime2, startcublas, stopcublas));
    printf("Bit flipping took %fms\n", cublasTime2);
	printf("Total time: %fms\n", cublasTime1 + cublasTime2);
	cudaErrCheck(cudaDeviceSynchronize());

	/* copy result back to host */
	/* fix the parameters needed to copy data back to the host */
	cudaErrCheck(cudaMemcpy(h_c, d_g, size, cudaMemcpyDeviceToHost ));



    printf("\n------------Result of  SELECT operation --Printing top 15 results------------\n");
	for (int i = 0; i < 16; i++) {
       int v2 = h_c[i];
       //printf("%d ",v2);
       //printf("%f %f %f %f\n", v2*original_table[i] , v2*original_table[i+4], v2*original_table[i+8], v2*original_table[i+12]);
      }

	/* clean up */

	cudaFree( d_g );

    //------------------------------------
   
      printf("\n\n------------Results verified: cublas------------\n\n");
      float cublasTime;
      //cudaDeviceSynchronize - CPU to wait until the kernel is done before it accesses the results (because CUDA kernel launches don’t block the calling CPU thread
      cudaErrCheck(cudaEventSynchronize(stopcublas));
      cudaErrCheck(cudaEventElapsedTime(&cublasTime, startcublas, stopcublas));
      printf("cublas took %fms\n", cublasTime);

   cudaErrCheck(cudaEventDestroy(startcublas));             
   cudaErrCheck(cudaEventDestroy(stopcublas));
   
   //free the data
   cudaErrCheck(cudaFree(d_a));
   cudaErrCheck(cudaFree(d_b));
   cudaErrCheck(cudaFree(d_c));
   
   cudaErrCheck(cudaDeviceReset());
   return 0;
}


