#include <iostream>
#include <cmath>


// Host: CPU, device: GPU
// CUDA Kernel function to add the elements of two arrays on the GPU
__global__
void add(int n, float* x, float* y)
{
    int index = threadIdx.x;
    int stride = blockDim.x;
    for (int i = index; i < n; i += stride) { y[i] = x[i] + y[i]; }
}

int main()
{
    int N = 1 << 20;

    float *x, *y;
    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Run kernel on 1M elements on the CPU
    // <<<number of thread blocks, number of threads in a thread block>>>
    add <<<1, 256>>> (N, x, y);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++) {
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    }

    std::cout << "Max eror: " << maxError << std::endl;

    // Free memory
    cudaFree(x);
    cudaFree(y);
}