
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


#define MATRIX_M 16
#define MATRIX_N 16
#define MATRIX_K 16

__global__ void condition_equal(float *in, float *out)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(in[index] == 2)  {
    out[index] = 1;
  }
  else {
    out[index] = 0;
  }
}

__global__ void condition_greaterthan(float *in, float *out)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(in[index] > 2)  {
    out[index] = 1;
  }
  else {
    out[index] = 0;
  }
}

__global__ void condition_lessthan(float *in, float *out)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(in[index] < 2)  {
    out[index] = 1;
  }
  else {
    out[index] = 0;
  }
}

int main(int argc, char* argv[]) {

   int condition;
   cublasHandle_t cublasHandle;
   
   cudaEvent_t startcublas;
   cudaEvent_t stopcublas;

   cudaErrCheck(cudaEventCreate(&startcublas));
   cudaErrCheck(cudaEventCreate(&stopcublas));
   
   cublasErrCheck(cublasCreate(&cublasHandle));
   

    printf("\n");
    printf("\n------------Original table ------------\n");
    
   float *original_table = new float[MATRIX_M*MATRIX_N];
   for(int i=0; i<MATRIX_M*MATRIX_N;i++){
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
   float *h_a = new float[MATRIX_M*MATRIX_N];
   float *h_c = new float[MATRIX_M*MATRIX_N];

   //TODO - create a array with initial values
    for(int i=0; i<MATRIX_M*MATRIX_N;i++){
        h_a[i] = 0;
        h_c[i] = 0;
    }
    
    for(int i=0;i<MATRIX_M;i++) {
        h_a[i] = original_table[i];
    }

    printf("\n");
    printf("Starting select operation, SELECT * FROM table WHERE A ? 2 ");
    printf("\n\n Enter 1 for equality condition\n 2 for less than condition\n 3 for greater than condition\n");
    scanf("%d", &condition);
    printf("\n");


    //allocate memory and copy the values from CPU to GPU
   float *d_a, *d_c;
    float  *m;
    float size = MATRIX_M*MATRIX_N * sizeof( float );
    m = (float *)malloc( size );
   cudaErrCheck(cudaMalloc((void**)&d_a, MATRIX_M * MATRIX_K * sizeof(float)));
   cudaErrCheck(cudaMalloc((void**)&d_c, MATRIX_K * MATRIX_N * sizeof(float)));


   cudaErrCheck(cudaMemcpy(d_a, h_a, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyHostToDevice));
   cudaErrCheck(cudaMemcpy(d_c, h_c, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyHostToDevice));
   
  	cudaErrCheck(cudaEventRecord(startcublas));

	  if(condition == 1)
		  condition_equal<<< 1, MATRIX_M>>>(d_a,d_c);
	  else if(condition == 2)
		condition_lessthan<<< 1, MATRIX_M>>>(d_a,d_c);
	  else
		 condition_greaterthan<<< 1, MATRIX_M>>>(d_a,d_c);

    float cublasTime;
	cudaErrCheck(cudaEventRecord(stopcublas));
     //cudaDeviceSynchronize - CPU to wait until the kernel is done before it accesses the results (because CUDA kernel launches don’t block the calling CPU thread
      cudaErrCheck(cudaEventSynchronize(stopcublas));
      cudaErrCheck(cudaEventElapsedTime(&cublasTime, startcublas, stopcublas));
      printf("select in gpu took %fms\n", cublasTime);
    cudaErrCheck(cudaDeviceSynchronize());

	/* copy result back to host */
	/* fix the parameters needed to copy data back to the host */
	cudaErrCheck(cudaMemcpy(m, d_c, size, cudaMemcpyDeviceToHost ));

    

     printf("\n------------Result of  SELECT operation --Printing top 15 results------------\n");
	for (int i = 0; i < 16; i++) {
       int v2 = m[i];
       //printf("%d ",v2);
       //printf("%f %f %f %f\n", v2*original_table[i] , v2*original_table[i+4], v2*original_table[i+8], v2*original_table[i+12]);
    }

	/* clean up */

	free(m);

    //------------------------------------
   
      printf("\n\n------------Results verified: Select in GPU ------------\n\n");
      //cudaDeviceSynchronize - CPU to wait until the kernel is done before it accesses the results (because CUDA kernel launches don’t block the calling CPU thread
      cudaErrCheck(cudaEventSynchronize(stopcublas));
      cudaErrCheck(cudaEventElapsedTime(&cublasTime, startcublas, stopcublas));
      printf("select in gpu took %fms\n", cublasTime);

   cudaErrCheck(cudaEventDestroy(startcublas));             
   cudaErrCheck(cudaEventDestroy(stopcublas));
   
   //free the data
   cudaErrCheck(cudaFree(d_a));
   cudaErrCheck(cudaFree(d_c));
   

   cudaErrCheck(cudaDeviceReset());
   return 0;
}


