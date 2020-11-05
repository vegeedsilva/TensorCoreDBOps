
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

__global__ void condition_equal(float *in, float *out, int selectCondition)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(in[index] == selectCondition)  {
    out[index] = 1;
  }
  else {
    out[index] = 0;
  }
}

__global__ void condition_greaterthan(float *in, float *out, int selectCondition)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(in[index] > selectCondition)  {
    out[index] = 1;
  }
  else {
    out[index] = 0;
  }
}

__global__ void condition_lessthan(float *in, float *out, int selectCondition)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(in[index] < selectCondition)  {
    out[index] = 1;
  }
  else {
    out[index] = 0;
  }
}

int main(int argc, char* argv[]) {

   int condition = 1;
   cublasHandle_t cublasHandle;
   cudaEvent_t startcublas;
   cudaEvent_t stopcublas;

   int selectCondition = atoi(argv[2]);

   long long MATRIX_M, MATRIX_N, MATRIX_K;
   MATRIX_M = atoi(argv[1]);
//    printf("Enter M size");
//    scanf("%lld", &MATRIX_M);

   MATRIX_N = MATRIX_M;
   MATRIX_K = MATRIX_M;

   cudaErrCheck(cudaEventCreate(&startcublas));
   cudaErrCheck(cudaEventCreate(&stopcublas));
   cublasErrCheck(cublasCreate(&cublasHandle));

   float *original_table = new float[MATRIX_M];
   for(int i=0; i<MATRIX_M;i++){
            original_table[i] = i;
            /*
            if(i%MATRIX_M==0)
                printf("Column %d printing ---->", (i/MATRIX_M)+1);
            printf(" %d", i);
            if((i+1)%MATRIX_M == 0){ 
            printf("\n"); 
             }*/
            
    }

   //create arrays with values
   float *h_a = new float[MATRIX_M];
   float *h_c = new float[MATRIX_M];

   //TODO - create a array with initial values
    for(int i=0; i<MATRIX_M;i++){
        h_a[i] = 0;
        h_c[i] = 0;
    }
    
    for(int i=0;i<MATRIX_M;i++) {
        h_a[i] = original_table[i];
    }

    //allocate memory and copy the values from CPU to GPU
   float *d_a, *d_c;
    float  *m;
    float size = MATRIX_M* sizeof( float );
    m = (float *)malloc( size );
   cudaErrCheck(cudaMalloc((void**)&d_a, MATRIX_M  * sizeof(float)));
   cudaErrCheck(cudaMalloc((void**)&d_c, MATRIX_K  * sizeof(float)));


   cudaErrCheck(cudaMemcpy(d_a, h_a, MATRIX_M * sizeof(float), cudaMemcpyHostToDevice));
   cudaErrCheck(cudaMemcpy(d_c, h_c, MATRIX_M * sizeof(float), cudaMemcpyHostToDevice));
   
  	cudaErrCheck(cudaEventRecord(startcublas));

	  if(condition == 1)
      condition_equal<<< (MATRIX_M + 255) / 256, 256 >>>(d_a, d_c, selectCondition);
	  else if(condition == 2)
		condition_lessthan<<<  (MATRIX_M + 255) / 256, 256 >>>(d_a,d_c, selectCondition);
	  else
		 condition_greaterthan<<<  (MATRIX_M + 255) / 256, 256 >>>(d_a,d_c, selectCondition);

    float cublasTime;
	cudaErrCheck(cudaEventRecord(stopcublas));
     //cudaDeviceSynchronize - CPU to wait until the kernel is done before it accesses the results (because CUDA kernel launches don’t block the calling CPU thread
      cudaErrCheck(cudaEventSynchronize(stopcublas));
      cudaErrCheck(cudaEventElapsedTime(&cublasTime, startcublas, stopcublas));
      printf("%fms", cublasTime);
    cudaErrCheck(cudaDeviceSynchronize());

	/* copy result back to host */
	/* fix the parameters needed to copy data back to the host */
	cudaErrCheck(cudaMemcpy(m, d_c, size, cudaMemcpyDeviceToHost ));

	for (int i = 0; i < 16; i++) {
       int v2 = m[i];
       //printf("%d ",v2);
       //printf("%f %f %f %f\n", v2*original_table[i] , v2*original_table[i+4], v2*original_table[i+8], v2*original_table[i+12]);
    }

	/* clean up */

	free(m);

    //------------------------------------

      //cudaDeviceSynchronize - CPU to wait until the kernel is done before it accesses the results (because CUDA kernel launches don’t block the calling CPU thread
      cudaErrCheck(cudaEventSynchronize(stopcublas));
      cudaErrCheck(cudaEventElapsedTime(&cublasTime, startcublas, stopcublas));

   cudaErrCheck(cudaEventDestroy(startcublas));             
   cudaErrCheck(cudaEventDestroy(stopcublas));
   
   //free the data
   cudaErrCheck(cudaFree(d_a));
   cudaErrCheck(cudaFree(d_c));
   

   cudaErrCheck(cudaDeviceReset());
   return 0;
}


