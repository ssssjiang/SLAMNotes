#include <cstdio>

__global__
void add(int n, float *x, float *y)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = index; i < n; i++) {
        y[i] = x[i] + y[i];
    }
}

int main()
{
    int N = 20 * (1 << 20);

    float *x = new float [N];
    float *y = new float [N];

    for (int i = 0; i < N; i++) {
        x[i]  =1.0f;
        y[i] = 2.0f;
    }

    float *d_x, *d_y;
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));

    cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 512;
    int blockNum = (N - blockSize + 1) / blockSize;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    add<<<blockNum, blockSize>>>(N, d_x, d_y);
    cudaEventRecord(stop);

    cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    float maxError = 0.0f;
    for (int i = 0; i < N; i++) {
        maxError = max(maxError, abs(y[i] - 4.0f));
    }

    printf("Max error: %fn", maxError);
    printf("Effective Bandwidth (GB/s): %fn", N * 4 * 3 / milliseconds / 1e6);

    delete [] x;
    delete [] y;
    cudaFree(d_x);
    cudaFree(d_y);
}