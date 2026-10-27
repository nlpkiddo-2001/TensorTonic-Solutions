#include <cuda_runtime.h>
#include <algorithm>

__global__ void relu_kernel(const float* input, float* output, int N) {
    // Store N element-wise results in output; this kernel returns no value.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N){
        output[idx] = fmaxf(input[idx], 0); 
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    relu_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}