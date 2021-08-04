#include <cstdio>

__global__
void saxpy(int n, float a, float *x, float *y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

int main()
{
    int N = 1 << 20;
    float *x = new float [N];
    float *y = new float [N];

    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    float *dx, *dy;
    cudaMallocManaged(&dx, N * sizeof(float));
    cudaMallocManaged(&dy, N * sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMemcpy(dx, x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dy, y, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(start);
    saxpy<<<(N + 255)/ 256, 256>>>(N, 2.0f, dx, dy);
    // cudaDeviceSynchronize();
    cudaEventRecord(stop);

    cudaMemcpy(y, dy, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);


    float maxError = 0.0f;
    for (int i = 0; i < N; i++) { maxError = max(maxError, abs(y[i] - 4.0f)); }
    
    printf("Max error: %f\n", maxError);
    printf("Effective BandWidth (GB/s): %fn", N * 4 * 3 / milliseconds / 1e6);

    cudaFree(dx);
    cudaFree(dy);
    free(x);
    free(y);
}