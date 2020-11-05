
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

__global__ void condition_equal(float *table1, float *table2, float *out, long MATRIX_M, long MATRIX_N)
{
  int index1 = blockIdx.x * blockDim.x + threadIdx.x;
  int index2= blockIdx.y * blockDim.y + threadIdx.y;
  if (index1>=MATRIX_M || index2>=MATRIX_N)
        return;
 
  if(table1[index1] == table2[index2]) {
    out[index1*MATRIX_M+index2] = 1;
  } else {
      out[index1*MATRIX_M+index2] =  0;
  }
}

int main(int argc, char* argv[]) {
   long long MATRIX_M, MATRIX_N, MATRIX_K;
   long long size1= atoi(argv[1]); 
 
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
   //cublasErrCheck(cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH));

   //malloc - void *malloc(size_t size) allocates the requested memory and returns a pointer to it.c function
   c_host_cublas = (float*)malloc(MATRIX_M * MATRIX_N * sizeof(float));

    //printf("\n");
    //printf("\n------------Table 1------------\n");
    
    float *table1 = new float[MATRIX_M*MATRIX_N];
    float *table2 = new float[MATRIX_M*MATRIX_N];

    for(int i=1; i<=MATRIX_M*MATRIX_N;i++){
            table1[i-1] = i;
             
    }
  
   //printf("\n------------Table 2 ------------\n");
   for(int i=1; i<=MATRIX_M*MATRIX_N;i++){
            table2[i-1] = i;
            
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
        h_a[i] = table1[i];
        h_b[i] = table2[i];
    }

    //printf("\n");
    //printf("Starting join operation, SELECT * FROM table1 NATURAL JOIN table2;");
    //printf("\nM = %d, N = %d, K = %d.", MATRIX_M, MATRIX_N, MATRIX_K);


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
   
  
   convertFp32ToFp16 <<< (MATRIX_M * MATRIX_K + 255) / 256, MATRIX_M*MATRIX_N >>> (d_a_half, d_a, MATRIX_M * MATRIX_N);
   convertFp32ToFp16 <<< (MATRIX_M * MATRIX_K + 255) / 256, MATRIX_M*MATRIX_N >>> (d_b_half, d_b, MATRIX_M * MATRIX_N);


    float  *m;
	float *d_g;
	float size = MATRIX_M*MATRIX_N * sizeof( float );
	m = (float *)malloc( size );

	cudaErrCheck(cudaMalloc( (void **) &d_g, size ));
    cudaErrCheck(cudaMemcpy( d_g, m, size, cudaMemcpyHostToDevice ));

	float time;
    cudaErrCheck(cudaEventRecord(startcublas));

	/* launch the kernel on the GPU */
	/* insert the launch parameters to launch the kernel properly using blocks and threads */ 
	  dim3 block(MATRIX_M,MATRIX_N);
	  dim3 grid ((MATRIX_M+MATRIX_N-1)/MATRIX_M, (MATRIX_M+MATRIX_N-1)/MATRIX_M  );


	condition_equal<<< grid, block >>>(d_a, d_b, d_g, MATRIX_M, MATRIX_N);

	cudaErrCheck(cudaEventRecord(stopcublas));
	cudaErrCheck(cudaEventSynchronize(stopcublas));

	cudaErrCheck(cudaEventElapsedTime(&time, startcublas, stopcublas));
	cudaErrCheck(cudaDeviceSynchronize());

	printf("%fms", time);


	/* copy result back to host */
	/* fix the parameters needed to copy data back to the host */
	cudaErrCheck(cudaMemcpy(m, d_g, size, cudaMemcpyDeviceToHost ));


     //printf("\n------------Result of JOIN operation ------------\n");
	for (int i = 0; i < MATRIX_M*MATRIX_N; i++) {
       int v2 = m[i];
       //printf("%d ", v2);
       //if(v2==1)
      //printf("Merge Row %d of Table 1 and Row %d of Table 2 \n",(int)(i/4), i%4 );
        //printf("%f %f %f %f\n", v2*original_table[i] , v2*original_table[i+4], v2*original_table[i+8], v2*original_table[i+12]);
       }

	/* clean up */

	free(m);
	cudaFree( d_g );

    //------------------------------------
   
      //printf("\n\n------------Results verified: cublas------------\n\n");
      float cublasTime;
      //cudaDeviceSynchronize - CPU to wait until the kernel is done before it accesses the results (because CUDA kernel launches don’t block the calling CPU thread
      cudaErrCheck(cudaEventSynchronize(stopcublas));
      cudaErrCheck(cudaEventElapsedTime(&cublasTime, startcublas, stopcublas));
      //printf("cublas took %fms\n", cublasTime);

   cudaErrCheck(cudaEventDestroy(startcublas));             
   cudaErrCheck(cudaEventDestroy(stopcublas));
   
   //free the data
   cudaErrCheck(cudaFree(d_a));
   cudaErrCheck(cudaFree(d_b));
   cudaErrCheck(cudaFree(d_c));
   
   free(c_host_cublas);

   cudaErrCheck(cudaDeviceReset());
   return 0;
}