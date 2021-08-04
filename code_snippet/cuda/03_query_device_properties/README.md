# Chap 03 - How to Query Device Properties and Handle Errors in CUDA C/C++

## 1. 查询设备属性

```cudaGetDeviceCount()```可以获取设备数量，```cudaDeviceProp```结构支持CUDA设备属性查询。具体代码如下：
```C++
// query_device.cu

#include <cstdio>

int main()
{
    int n_devices;

    cudaGetDeviceCount(&n_devices);
    for (int i = 0; i < n_devices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device Name: %s\n", prop.name);
        printf("  Device Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n\n", 
               2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
    }
}
```
其中需要注意的是计算能力(Compute Capability)。```CUDA```的```major```和```minor```描述了设备的计算能力，通常以```major.minor```的形式给出。多处理器上的多线程块能够并发执行主要取决于可用的资源数量(on-chip registers and shared memory). 对于确定计算能力的设备，可以使用nvcc的编译器选项来生成代码```-arch=sm_xx```，```xx```指计算能力(没有小数点)。比如计算能力3.5的设备，可以设置选项```-arch=sm35```。

更具体的关于```cudaDeviceProp```的API可参考[cudaDeviceProp Struct Reference](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp)

## 处理CUDA错误

所有CUDA的Runtime API中的函数都有一个返回值，用于检查它们执行过程中发生的错误。执行成功的则返回```cudaSuccess```. ```cudaGetErrorString()```可以获取对于错误的描述.

处理内核错误比较复杂，因为内核线程是异步执行的。为了检查内核执行错误，CUDA的runtime维护一个错误变量，每当错误发生的时候，这个错误变量都会被覆写掉。```cudaGetLastError()```返回这个变量并且将它重置为```cudaSuccess```。具体代码如下
```C++
// handle_errors.cu

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
```

## Reference

1. https://devblogs.nvidia.com/how-query-device-properties-and-handle-errors-cuda-cc