There are 5 CUDA files for each database operation

1. **Select Operation**

- select-tensor.cu : This file runs the select where operation in tensor cores

- select-tensor-chunking.cu : This file runs the select where operation in tensor cores along with chunking for large input sizes

- select-without-tensor.cu : This file runs the select where operation without using tensor cores

- select-without-tensor-chunking.cu : This file runs the select where operation without using tensor cores and performs chunking for large input sizes

- select-baseline.cu: This file runs the select where operation in GPU and it is the baseline for our project

2. **Join Operation**

- join-tensor.cu : This file runs the join operation in tensor cores

- join-tensor-chunking.cu : This file runs the join operation in tensor cores along with chunking for large input sizes

- join-without-tensor.cu : This file runs the join operation without using tensor cores

- join-without-tensor-chunking.cu : This file runs the join operation without using tensor cores and performs chunking for large input sizes

- join-baseline.cu: This file runs the join operation in GPU and it is the baseline for our project

**Compilation and Execution**
The above mentioned CUDA files can be compiled as mentioned below.

**Compile**

/usr/local/cuda/bin/nvcc -o select-baseline -lcublas select-baseline.cu

**Execution**

/usr/local/cuda/bin/nvprof --unified-memory-profiling off ./select-baseline

nvprof is the NVIDIA Profiler used to print execution time taken for various operations.



