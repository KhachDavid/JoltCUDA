#include <cuda_runtime.h>
#include <stdio.h>

// CUDA Kernel to print "Hello, World!" on GPU
__global__ void HelloWorldKernel() {
    printf("Hello, hi from CUDA! Thread ID: %d\n", threadIdx.x);
}

// Function to launch the kernel
extern "C" void LaunchHelloKernel() {
    HelloWorldKernel<<<1, 10>>>();  // Launch with 10 threads
    cudaDeviceSynchronize();        // Ensure the kernel completes execution
}
