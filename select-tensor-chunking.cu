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

#define MATRIX_M 16
#define MATRIX_N 16
#define MATRIX_K 16
#define CHUNK_SIZE 1024

__global__ void convertFp32ToFp16 (half *out, float *in, int n,int start, int end) {
   int idx = blockDim.x * blockIdx.x + threadIdx.x;
   if ((idx + start) < end) {
      out[idx] = in[idx+start];
   }
}

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
    
   float *c_host_cublas;
   int condition;
   cublasHandle_t cublasHandle;
   float cublasTime2, cublasTime1;
   
   cudaEvent_t startcublas;
   cudaEvent_t stopcublas;

   cudaErrCheck(cudaEventCreate(&startcublas));
   cudaErrCheck(cudaEventCreate(&stopcublas));
   
   cublasErrCheck(cublasCreate(&cublasHandle));
   
   // Use tensor cores
   cublasErrCheck(cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH));

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
   
   
    if(MATRIX_M > CHUNK_SIZE){
      
          //create arrays with values
          float *h_a = new float[MATRIX_M*MATRIX_N];
          float *h_b = new float[MATRIX_M*MATRIX_N];
          float *h_c = new float[CHUNK_SIZE * CHUNK_SIZE ];

          //TODO - create a array with initial values
            for(int i=0; i<MATRIX_M*MATRIX_N;i++){
                h_a[i] = 0;
                h_b[i] = 0;
            }

            for(int i=0; i<CHUNK_SIZE*CHUNK_SIZE;i++){
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
          
          half *h_a_half = new half[CHUNK_SIZE*CHUNK_SIZE];
          half *h_b_half = new half[CHUNK_SIZE*CHUNK_SIZE];


          cudaErrCheck(cudaMalloc((void**)&d_a, MATRIX_M * MATRIX_K * sizeof(float)));
          cudaErrCheck(cudaMalloc((void**)&d_b, MATRIX_K * MATRIX_N * sizeof(float)));
          cudaErrCheck(cudaMalloc((void**)&d_c, MATRIX_K * MATRIX_N * sizeof(float)));
          cudaErrCheck(cudaMalloc((void**)&d_a_half, MATRIX_K * MATRIX_N * sizeof(half)));
          cudaErrCheck(cudaMalloc((void**)&d_b_half, MATRIX_K * MATRIX_N * sizeof(half)));

          cudaErrCheck(cudaMemcpy(d_a, h_a, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyHostToDevice));
          cudaErrCheck(cudaMemcpy(d_b, h_b, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyHostToDevice));
          cudaErrCheck(cudaMemcpy(d_a_half, h_a_half, CHUNK_SIZE * CHUNK_SIZE * sizeof(half), cudaMemcpyHostToDevice));
          cudaErrCheck(cudaMemcpy(d_b_half, h_b_half, CHUNK_SIZE * CHUNK_SIZE * sizeof(half), cudaMemcpyHostToDevice));
          cudaErrCheck(cudaMemcpy(d_c, h_c, CHUNK_SIZE * CHUNK_SIZE * sizeof(float), cudaMemcpyHostToDevice));

          float *d_g;
          float size = MATRIX_M*MATRIX_N * sizeof( float );
          
          cudaErrCheck(cudaMalloc( (void **) &d_g, size ));
          cudaErrCheck(cudaMemcpy( d_g, h_a, size, cudaMemcpyHostToDevice ));
          
          printf("------------Running with cuBLAS------------n");
          float totaltime = 0;
          int noOfLoops = MATRIX_M/CHUNK_SIZE;
          for(int i =0; i < noOfLoops; i++){
                int start = i*CHUNK_SIZE;
                int end = start + CHUNK_SIZE;
                convertFp32ToFp16 <<< (MATRIX_M * MATRIX_K + 255) / 256, 256 >>> (d_a_half, d_a, MATRIX_M * MATRIX_N, start, end);
                convertFp32ToFp16 <<< (MATRIX_M * MATRIX_K + 255) / 256, 256  >>> (d_b_half, d_b, MATRIX_M * MATRIX_N, start, end);

                float alpha = 1.0f;
                float beta = 1.0f;

                printf("\nM = %d, N = %d, K = %d. alpha = %f, beta = %f\n\n", MATRIX_M, MATRIX_N, MATRIX_K, alpha, beta);

                float cublasTime1;
                float cublasTime2;

                // Now using cuBLAS
                cudaErrCheck(cudaEventRecord(startcublas));
                cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
                              CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE,
                              &alpha,
                              d_a_half, CUDA_R_16F, CHUNK_SIZE,
                              d_b_half, CUDA_R_16F, CHUNK_SIZE,
                              &beta, 
                              d_c, CUDA_R_32F, CHUNK_SIZE,
                              CUDA_R_32F,  CUBLAS_GEMM_DFALT_TENSOR_OP));


                  /* launch the kernel on the GPU */
                  /* insert the launch parameters to launch the kernel properly using blocks and threads */ 
                 if(condition == 1)
                   condition_equal<<< 1, CHUNK_SIZE >>>(d_c,d_g, start);
                 else if(condition == 2)
					condition_lessthan<<< 1, CHUNK_SIZE >>>(d_c,d_g, start);
                 else
					condition_greaterthan<<< 1, CHUNK_SIZE >>>(d_c,d_g, start);
                 cudaErrCheck(cudaEventRecord(stopcublas));
                 cudaErrCheck(cudaEventSynchronize(stopcublas));
                 cudaErrCheck(cudaEventElapsedTime(&cublasTime1, startcublas, stopcublas));
                 totaltime += cublasTime1;
                 printf("Matrix multiplication %d took %fms\n", i+1, cublasTime1);

                 cudaErrCheck(cudaDeviceSynchronize());

          }
          printf("Matrix multiplication total time %fms", totaltime);

          /* copy result back to host */
          /* fix the parameters needed to copy data back to the host */
          cudaErrCheck(cudaMemcpy(h_b, d_g, size, cudaMemcpyDeviceToHost ));

   
          printf("\n------------Result of  SELECT operation - Printing first 20 results------------\n");
          for (int i = 0; i < 20; i++) {
              //printf("%f ",h_b[i]);
              //printf("%f %f %f %f\n", v2*original_table[i] , v2*original_table[i+4], v2*original_table[i+8], v2*original_table[i+12]);
          }
		  

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

          float *d_g;
          float size = MATRIX_M*MATRIX_N * sizeof( float );
          
          cudaErrCheck(cudaMalloc( (void **) &d_g, size ));
          cudaErrCheck(cudaMemcpy( d_g, h_a, size, cudaMemcpyHostToDevice ));
          
          printf("------------Running with cuBLAS------------n");
          float totaltime=0;
          int start = 0;
          int end = MATRIX_M*MATRIX_N;

          convertFp32ToFp16 <<< (MATRIX_M * MATRIX_K + 255) / 256, 256 >>> (d_a_half, d_a, MATRIX_M * MATRIX_N, start, end);
          convertFp32ToFp16 <<< (MATRIX_M * MATRIX_K + 255) / 256, 256 >>> (d_b_half, d_b, MATRIX_M * MATRIX_N, start, end);

         

          float alpha = 1.0f;
          float beta = 1.0f;

          printf("\nM = %d, N = %d, K = %d. alpha = %f, beta = %f\n\n", MATRIX_M, MATRIX_N, MATRIX_K, alpha, beta);

          float cublasTime1;
          float cublasTime2;

          // Now using cuBLAS
          cudaErrCheck(cudaEventRecord(startcublas));
          cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
                        MATRIX_M, MATRIX_N, MATRIX_K,
                        &alpha,
                        d_a_half, CUDA_R_16F, MATRIX_M,
                        d_b_half, CUDA_R_16F, MATRIX_N,
                        &beta, 
                        d_c, CUDA_R_32F, MATRIX_K,
                        CUDA_R_32F,  CUBLAS_GEMM_DFALT_TENSOR_OP));


            /* launch the kernel on the GPU */
            /* insert the launch parameters to launch the kernel properly using blocks and threads */ 
           if(condition == 1)
              condition_equal<<< 1, MATRIX_M >>>(d_c,d_g, start);
           else if(condition == 2)
				   	  condition_lessthan<<< 1, MATRIX_M >>>(d_c,d_g, start);
           else
					    condition_greaterthan<<< 1, MATRIX_M >>>(d_c,d_g, start);
            cudaErrCheck(cudaEventRecord(stopcublas));
            cudaErrCheck(cudaEventSynchronize(stopcublas));
            cudaErrCheck(cudaEventElapsedTime(&cublasTime1, startcublas, stopcublas));
            totaltime = cublasTime1;
            printf("Matrix multiplication %d took %fms\n", 1, cublasTime1);
            cudaErrCheck(cudaDeviceSynchronize());

          printf("Matrix multiplication total time %fms", totaltime);

          /* copy result back to host */
          /* fix the parameters needed to copy data back to the host */
          cudaErrCheck(cudaMemcpy(h_c, d_g, size, cudaMemcpyDeviceToHost ));

   
          printf("\n------------Result of  SELECT operation - Printing first 15 results ------------\n");
          for (int i = 0; i < 16; i++) {
              //printf("%f ", h_c[i]);

              //printf("%f %f %f %f\n", v2*original_table[i] , v2*original_table[i+4], v2*original_table[i+8], v2*original_table[i+12]);
          }
		  

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