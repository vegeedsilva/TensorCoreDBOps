#include <stdio.h>
#include <curand.h>
#include <cublas_v2.h>
#include <vector>
#include <math.h>
#include <cmath>

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

//CONVERSION TO HALF PRECISION TO SUPPORT TENSOR CORES
__global__ void convertFp32ToFp16 (half *out, float *in, int n,int start, int end) {
   int idx = blockDim.x * blockIdx.x + threadIdx.x;
   if ((idx + start) < end) {
      out[idx] = in[idx+start];
   }
}

//BIT FLIPPING
__global__ void condition_equal(float *in, float *out, int start)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
    out[index+start] = (((int)ceil(abs(in[index]-1)))&&1) ^ 1;
}

__global__ void condition_greaterthan(float *in, float *out,int start)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
    out[index+start] = ((int)(ceil(in[index]-1)+ abs(ceil(in[index]-1)))) && 1;
}

__global__ void condition_lessthan(float *in, float *out, int start)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
    out[index+start] = ((int)(floor(in[index]-1) - abs(floor(in[index]-1)))) && 1;
}

int main(int argc, char* argv[]) {	

   int condition = 1; //selection condition default to 1 (Equality condition)
   long long size = atoi(argv[1]);
   int selectCondition = atoi(argv[4]);
//   printf("Enter matrix size in powers of two ");
//   scanf("%lld", &size);

//   printf("%lld\t", size);

   cublasHandle_t cublasHandle;
   long long MATRIX_M, MATRIX_N, MATRIX_K;
   long long CHUNK_MATRIX_SIZE = atoi(argv[2]); // chunk matrix is of dimension 2^20 i.e. 2^10 * 2^10 matrix
   long long CHUNK_MATRIX_M = CHUNK_MATRIX_SIZE * CHUNK_MATRIX_SIZE; // chunk size is taken as 2^20
   MATRIX_M = size;
   MATRIX_N = size;
   MATRIX_K = size;

   cudaEvent_t startcublas;
   cudaEvent_t stopCUDAGEMM;
   cudaEvent_t stopcublas;

   int THREAD_SIZE = atoi(argv[3]);
   cudaErrCheck(cudaEventCreate(&startcublas));
   cudaErrCheck(cudaEventCreate(&stopcublas));
  cudaErrCheck(cudaEventCreate(&stopCUDAGEMM));
   cublasErrCheck(cublasCreate(&cublasHandle));
   // Use tensor cores
   cublasErrCheck(cublasSetMathMode(cublasHandle, CUBLAS_DEFAULT_MATH));
//   printf("\n------------Original table ------------\n");
   float *original_table = new float[MATRIX_M];
/*   for(int i=0; i<MATRIX_M; i++){
            original_table[i] = 3;
            // if(i%MATRIX_M==0)
            //     printf("Column %d printing ---->", (i/MATRIX_M)+1);
            // printf(" %d", i);
            // if((i+1)%MATRIX_M == 0){
            // printf("\n");
            // }
    }

*/

    original_table[0] = 2; // input matrix is 2 3 3 ... So result should be 1 0 0 .. for equality condition

//    printf("Starting select operation, SELECT * FROM table WHERE A ? 2 ");
//    printf("\n\n Enter 1 for equality condition\n 2 for less than condition\n 3 for greater than condition\n");
//    scanf("%d", &condition); //uncomment to give other condition

    if(MATRIX_M > CHUNK_MATRIX_M){
          long long original_size = MATRIX_M;
          long matrix_size = (long)ceil(sqrt(original_size));
          MATRIX_M = matrix_size*matrix_size;
          
          //create arrays with values
          float *h_a =(float*)calloc(MATRIX_M, sizeof(float));
          float *h_b =(float*)calloc(MATRIX_N, sizeof(float));
          float *h_c =(float*)calloc(CHUNK_MATRIX_M, sizeof(float));
     
          for(long long i=0;i<original_size;i++) {
              h_a[i] = original_table[i];
          }
          
          for(long long i=0; i<MATRIX_M; i = i + (matrix_size) + 1) { 
              h_b[i] = 1/selectCondition;
          }
          
          //printf("\n");
          //allocate memory and copy the values from CPU to GPU
          float *d_a, *d_b, *d_c;
          half *d_a_half, *d_b_half;
          
          half *h_a_half = new half[CHUNK_MATRIX_M];
          half *h_b_half = new half[CHUNK_MATRIX_M];

          cudaErrCheck(cudaMalloc((void**)&d_a, MATRIX_M * sizeof(float)));
          cudaErrCheck(cudaMalloc((void**)&d_b, MATRIX_M * sizeof(float)));
          cudaErrCheck(cudaMalloc((void**)&d_c, MATRIX_M * sizeof(float)));
          cudaErrCheck(cudaMalloc((void**)&d_a_half, MATRIX_M * sizeof(half)));
          cudaErrCheck(cudaMalloc((void**)&d_b_half, MATRIX_M * sizeof(half)));

          cudaErrCheck(cudaMemcpy(d_a, h_a, MATRIX_M * sizeof(float), cudaMemcpyHostToDevice));
          cudaErrCheck(cudaMemcpy(d_b, h_b, MATRIX_M * sizeof(float), cudaMemcpyHostToDevice));
          cudaErrCheck(cudaMemcpy(d_a_half, h_a_half, CHUNK_MATRIX_M * sizeof(half), cudaMemcpyHostToDevice));
          cudaErrCheck(cudaMemcpy(d_b_half, h_b_half, CHUNK_MATRIX_M * sizeof(half), cudaMemcpyHostToDevice));
          cudaErrCheck(cudaMemcpy(d_c, h_c, CHUNK_MATRIX_M *  sizeof(float), cudaMemcpyHostToDevice));

          float *d_g;
          float size = MATRIX_M * sizeof( float );
          
          cudaErrCheck(cudaMalloc( (void **) &d_g, size ));
          cudaErrCheck(cudaMemcpy( d_g, h_a, size, cudaMemcpyHostToDevice ));
          
//          printf("------------Running with cuBLAS------------n");
          float totaltime = 0;
	       float totalCUDA = 0;
          int noOfLoops = MATRIX_M/CHUNK_MATRIX_M;
          for(int i =0; i < noOfLoops; i++){
                int start = i*CHUNK_MATRIX_M;
                int end = start + CHUNK_MATRIX_M;
                convertFp32ToFp16 <<< (MATRIX_M + THREAD_SIZE-1) / THREAD_SIZE, THREAD_SIZE  >>> (d_a_half, d_a, MATRIX_M , start, end);
                convertFp32ToFp16 <<< (MATRIX_M + THREAD_SIZE-1) / THREAD_SIZE, THREAD_SIZE   >>> (d_b_half, d_b, MATRIX_M, start, end);

                float alpha = 1.0f;
                float beta = 0.0f;

  //              printf("\nM = %lld, N = %lld, K = %lld. alpha = %f, beta = %f\n\n", MATRIX_M, MATRIX_N, MATRIX_K, alpha, beta);

                float cublasTime1;
                // Now using cuBLAS
                cudaErrCheck(cudaEventRecord(startcublas));
                cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                              CHUNK_MATRIX_SIZE, CHUNK_MATRIX_SIZE, CHUNK_MATRIX_SIZE,
                              &alpha,
                              d_a, CUDA_R_32F, CHUNK_MATRIX_SIZE,
                              d_b, CUDA_R_32F, CHUNK_MATRIX_SIZE,
                              &beta,
                              d_c, CUDA_R_32F, CHUNK_MATRIX_SIZE,
                              CUDA_R_32F, CUBLAS_GEMM_DEFAULT));
                  cudaErrCheck(cudaEventRecord(stopCUDAGEMM));
                  cudaErrCheck(cudaEventSynchronize(stopCUDAGEMM));
                  cudaErrCheck(cudaEventElapsedTime(&cublasTime1, startcublas, stopCUDAGEMM));
                  totalCUDA += cublasTime1;

                  /* launch the kernel on the GPU */
                  /* insert the launch parameters to launch the kernel properly using blocks and threads */
                 if(condition == 1)
                   condition_equal<<< (CHUNK_MATRIX_SIZE * CHUNK_MATRIX_SIZE + THREAD_SIZE-1) / THREAD_SIZE, THREAD_SIZE  >>>(d_c,d_g, start); //Bala -- vary this
                 else if(condition == 2)
					          condition_lessthan<<< (CHUNK_MATRIX_SIZE * CHUNK_MATRIX_SIZE + THREAD_SIZE-1) / THREAD_SIZE, THREAD_SIZE   >>>(d_c,d_g, start);
                 else
					          condition_greaterthan<<< (CHUNK_MATRIX_SIZE * CHUNK_MATRIX_SIZE + THREAD_SIZE-1) / THREAD_SIZE, THREAD_SIZE  >>>(d_c,d_g, start);

                 cudaErrCheck(cudaEventRecord(stopcublas));
                 cudaErrCheck(cudaEventSynchronize(stopcublas));
                 cudaErrCheck(cudaEventElapsedTime(&cublasTime1, startcublas, stopcublas));

                 totaltime += cublasTime1;
//                 printf("%fms\n", cublasTime1);
//                 printf("Matrix multiplication %d took %fms\n", i+1, cublasTime1);

                 cudaErrCheck(cudaDeviceSynchronize());

          }
          printf("%fms", totaltime);
          /* copy result back to host */
          /* fix the parameters needed to copy data back to the host */
          cudaErrCheck(cudaMemcpy(h_b, d_g, size, cudaMemcpyDeviceToHost ));
/*          printf("\n------------Result of  SELECT operation - Printing first 20 results------------\n");
          for (int i = 0; i < 20; i++) {
              printf("%f ",h_b[i]);
              //printf("%f %f %f %f\n", v2*original_table[i] , v2*original_table[i+4], v2*original_table[i+8], v2*original_table[i+12]);
          }
*/		

          /* clean up */
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
    else{
          // The given matrix size should be a perfect square and a power of 2 
          // else pad to it the next matrix size which satisfies that condition
          long long original_size = MATRIX_M;
          long matrix_size = (long)ceil(sqrt(original_size));
        //   if(ceil(sqrt(original_size)) != (int)sqrt(original_size)) 
        //     matrix_size = (int)sqrt(pow(2, ceil(log(original_size +1 )/log(2))));
          MATRIX_M = matrix_size*matrix_size;

//          printf("Original -- Appended matrix %lld %lld  ", original_size, MATRIX_M);
          //create arrays with values
          float *h_a = new float[MATRIX_M];
          float *h_b = new float[MATRIX_M];
          float *h_c = new float[MATRIX_M];
          
          for(int i=0; i< MATRIX_M;i++){
            h_a[i] = 0;
            h_b[i] = 0;
            h_c[i] = 0;
         }
          for(int i=0;i< original_size;i++) {
                h_a[i] = original_table[i];
          }
          for(int i=0; i< MATRIX_M; i = i + (matrix_size) + 1) { 
                h_b[i] = 1/selectCondition;
          }

            // printf("\n------------Printing table A------------\n");
            // for(int i=0; i<200;i++){
            //     printf("%f ", h_a[i]);
            //     if((i+1) % matrix_size == 0){
            //         printf("\n");
            //     }
            // }

            // printf("\n------------Printing table B------------\n");
            // for(int i=0; i<200;i++){
            //     printf("%f ", h_b[i]);
            //     if((i+1) % matrix_size == 0){
            //         printf("\n");
            //     }
            // }


          //allocate memory and copy the values from CPU to GPU
          float *d_a, *d_b, *d_c;
          half *d_a_half, *d_b_half;
          
          half *h_a_half = new half[MATRIX_M];
          half *h_b_half = new half[MATRIX_M];


          cudaErrCheck(cudaMalloc((void**)&d_a, MATRIX_M * sizeof(float)));
          cudaErrCheck(cudaMalloc((void**)&d_b, MATRIX_M* sizeof(float)));
          cudaErrCheck(cudaMalloc((void**)&d_c, MATRIX_M * sizeof(float)));
          cudaErrCheck(cudaMalloc((void**)&d_a_half, MATRIX_M * sizeof(half)));
          cudaErrCheck(cudaMalloc((void**)&d_b_half, MATRIX_M * sizeof(half)));

          cudaErrCheck(cudaMemcpy(d_a, h_a, MATRIX_M * sizeof(float), cudaMemcpyHostToDevice));
          cudaErrCheck(cudaMemcpy(d_b, h_b, MATRIX_M * sizeof(float), cudaMemcpyHostToDevice));
          cudaErrCheck(cudaMemcpy(d_a_half, h_a_half, MATRIX_M * sizeof(half), cudaMemcpyHostToDevice));
          cudaErrCheck(cudaMemcpy(d_b_half, h_b_half, MATRIX_M * sizeof(half), cudaMemcpyHostToDevice));
          cudaErrCheck(cudaMemcpy(d_c, h_c, MATRIX_M * sizeof(float), cudaMemcpyHostToDevice));

          float *d_g;
          float size = MATRIX_M * sizeof( float );
          
          cudaErrCheck(cudaMalloc( (void **) &d_g, size ));
          cudaErrCheck(cudaMemcpy( d_g, h_a, size, cudaMemcpyHostToDevice ));
          
//          printf("------------Running with cuBLAS------------n");
          float totaltime=0;
          int start = 0;
          int end = MATRIX_M;

          convertFp32ToFp16 <<< (matrix_size*matrix_size + THREAD_SIZE-1) / THREAD_SIZE , THREAD_SIZE >>> (d_a_half, d_a, matrix_size* matrix_size, start, end);
          convertFp32ToFp16 <<< (matrix_size*matrix_size + THREAD_SIZE-1) / THREAD_SIZE , THREAD_SIZE >>> (d_b_half, d_b, matrix_size* matrix_size, start, end);

          float alpha = 1.0f;
          float beta = 0.0f;

//          printf("\nM = %lld, N = %lld, K = %lld. alpha = %f, beta = %f\n\n", MATRIX_M, MATRIX_N, MATRIX_K, alpha, beta);

          float cublasTime1;
          // Now using cuBLAS
          cudaErrCheck(cudaEventRecord(startcublas));
          cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
                        matrix_size, matrix_size, matrix_size,
                        &alpha,
                        d_a, CUDA_R_32F, matrix_size,
                        d_b, CUDA_R_32F, matrix_size,
                        &beta, 
                        d_c, CUDA_R_32F, matrix_size,
                        CUDA_R_32F,  CUBLAS_GEMM_DEFAULT));
	cudaErrCheck(cudaEventRecord(stopCUDAGEMM));
            /* launch the kernel on the GPU */
            /* insert the launch parameters to launch the kernel properly using blocks and threads */ 
           if(condition == 1)
              condition_equal<<< (matrix_size*matrix_size + THREAD_SIZE -1) / THREAD_SIZE , THREAD_SIZE  >>>(d_c,d_g, start);
           else if(condition == 2)
			        condition_lessthan<<< (matrix_size*matrix_size + THREAD_SIZE -1) / THREAD_SIZE , THREAD_SIZE  >>>(d_c,d_g, start);
           else
              condition_greaterthan<<< (matrix_size*matrix_size + THREAD_SIZE -1) / THREAD_SIZE , THREAD_SIZE >>>(d_c,d_g, start);
              
            cudaErrCheck(cudaEventRecord(stopcublas));
            cudaErrCheck(cudaEventSynchronize(stopcublas));
            cudaErrCheck(cudaEventElapsedTime(&cublasTime1, startcublas, stopcublas));
            totaltime = cublasTime1;
            cudaErrCheck(cudaDeviceSynchronize());
            printf("%fms", totaltime);

          /* copy result back to host */
          /* fix the parameters needed to copy data back to the host */
          cudaErrCheck(cudaMemcpy(h_c, d_g, size, cudaMemcpyDeviceToHost ));

   
/*          printf("\n------------Result of  SELECT operation - Printing first 15 results ------------\n");
          for (int i = 0; i < 16; i++) {
              printf("%f ", h_c[i]);
              //printf("%f %f %f %f\n", v2*original_table[i] , v2*original_table[i+4], v2*original_table[i+8], v2*original_table[i+12]);
          }
*/		  
          /* clean up */
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
}
