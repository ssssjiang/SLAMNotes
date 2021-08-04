#include <cstdio>

__global__
void add(int n, float *x, float *y)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = index; i < n; i += gridDim.x * blockDim.x) {
        y[i] = x[i] + y[i];
    }
}

int main()
{
    int N = 1 << 20;

    float *x, *y;
    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));

    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    int blockSize = 256;
    int blockNum = (N - blockSize + 1) / blockSize;
    add<<<blockNum, blockSize>>>(N, x, y);

    cudaError_t errSync = cudaGetLastError();
    cudaError_t errAsync = cudaDeviceSynchronize();
    if (errSync != cudaSuccess) {
        printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
    }
    if (errAsync != cudaSuccess) {
        printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
    }

    cudaDeviceSynchronize();

    cudaFree(x);
    cudaFree(y);
}