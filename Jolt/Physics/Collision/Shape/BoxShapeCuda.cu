#include <cuda_runtime.h>
#include <stdio.h>

// CUDA Kernel to print "Hello, World!" on GPU
__global__ void HelloWorldKernel() {
    printf("Hello, World from CUDA! Thread ID: %d\n", threadIdx.x);
}
