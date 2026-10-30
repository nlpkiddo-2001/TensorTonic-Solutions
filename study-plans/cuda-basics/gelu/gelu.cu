#include <cuda_runtime.h>
#include <math.h>

__global__ void gelu_kernel(const float* input, float* output, int N) {
    // Write the float32 results into output; this kernel returns no value.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = 0.5f * input[idx] * (1.0f + erff(input[idx] * 0.70710678118654752440f));
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    dim3 blocks((N + 255) / 256);
    gelu_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}
