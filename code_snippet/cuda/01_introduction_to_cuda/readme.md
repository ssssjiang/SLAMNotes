# Chapter 01 - An Even Easier Introduction to CUDA



## 1. 开始样例

首先从一个C++的示例代码开始
```C++
// add.cpp

#include <iostream>
#include <cmath>

// function to add the elements of two arrays
void add(int n, float *x, float *y)
{
    for (int i = 0; i < n; i++) { y[i] = x[i] + y[i]; }
}

int main()
{
    int N = (1 << 20); // 1M elements

    float *x = new float[N];
    float *y = new float[N];

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Run kernel on 1M elements on the CPU
    add(N, x, y);
    
    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++) {
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    }
    std::cout << "Max error: " << maxError << std::endl;

    // Free memory
    delete [] x;
    delete [] y;
}
```
接着，需要将C++代码修改成CUDA代码。

## 2. CUDA中的内存分配

首先，需要将```add()```函数修改成能够再GPU上运行的代码，在```CUDA```中，称为```核函数(kernel)```。为此，在```add```函数的前加入```__global__```，这等于告诉```CUDA```的C++编译器，这个函数将在```GPU```上运行并且能在```CPU```上调用。

为了能够使用GPU计算，需要分配能被GPU访问的内存。```CUDA```的
[Unified Memory](https://devblogs.nvidia.com/unified-memory-in-cuda-6)可以在你的系统上提供一个
GPU和CPU都能访问的内存空间。```cudaMallocManaged()```可以用于分配unified memory，```cudaFree()```用于释放
内存。因此，只需将对应的代码修改为
```C++
float *x, *y;
cudaMallocManaged(&x, N * sizeof(float));
cudaMallocManaged(&y, N * sizeof(float));
```
以及内存释放代码修改为
```C++
cudaFree(x);
cudaFree(y);
```

完整代码如下：
```C++
// add.cu

#include <iostream>
#include <cmath>


// Host: CPU, device: GPU
// CUDA Kernel function to add the elements of two arrays on the GPU
__global__
void add(int n, float* x, float* y)
{
    for (int i = 0; i < n; i++) { y[i] = x[i] + y[i]; }
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
```

## 3. 核函数调用
```CUDA```使用三个尖角括号来调用核函数```<<<>>>```。要想使用```add()```函数，那么，调用方式应该为
```add<<<blocksNum, threadsNum>>>```。```blocksNum```指的是线程块的数目，```threadsNum```指的是每个线程块中使用多少个线程。例如```add<<<1, 256>>>(N, x, y)```.

## 4. CUDA中的几个常用变量

```CUDA```的GPU中有许多并行处理器被分在了```Streaming Multiprocessors (SMs)```中。每个SM能够运行多个并发的线程块。比如基于Pascal GPU架构的Tesla P100，每个SM能够支持最多2048个活跃线程。

访问这些线程的几个基本概念(**一维的情况下**)：
- ```grid```： 可并行的线程块构成一个grid.
- ```threadIdx.x```： 当前线程所在线程块的索引.
- ```blockDim.x```： 线程块所拥有的线程数量.
- ```blockIdx.x```： 当前线程块所在grid的索引.
- ```gridDim.x```： grid中的线程块数量.

具体解释可见下图：
![CUDA线程块示意图](https://devblogs.nvidia.com/wp-content/uploads/2017/01/cuda_indexing.png)

### 4.1. 如何计算
首先我们需要知道如何计算核函数调用时所需的线程块的数量。在运行核函数时，CUDA的GPU使用的线程块的线程数是**32**的整数倍。因此，假设每个线程块，我们使用其中的 ```blockSize = 256``` 个线程，并且，我们有```N```个元素需要处理。所以，我们至少需要的线程块的数量为 ```numBlocks = (N + blockSize - 1) / blockSize``` 。

对于一维数组的访问也很简单，索引的计算方法为 ```index = blockIdx.x * blockDim.x + threadIdx.x``` . 示例代码：

```C++
// add_block.cu

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
```

### 4.2. 步长(stride)计算

```# monolithic kernel #```

常见的```CUDA```写法就是对每个数据元素建立一个线程，这个写法是建立在我们拥有足够的线程的情况下：
```C++
__global__ 
void add(int n, float *x, float *y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] + y[i];
}
```

```# grid-stride loop #```

现在假设我们没有足够多的线程来覆盖所有的数据项，我们就需要使用 ```grid-stride loop``` .
```C++
__global__
void add(int n, float *x, float *y)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = index; i < n; i += blockDim.x * gridDim.x) {
        y[i] = x[i] + y[i];
    }
}
```
循环的步长(stride)等于 ```blockDim.x * blockDim.x```，grid中的线程总数。因此，如果grid中总共有1280个线程，那么id为0的线程将会计算元素0,1280,2560等等。使用 ```grid-stride loop``` 有如下好处：
1. 可扩展性和线程复用(Scalability and thread reuse). 即使数据规模超过了CUDA设备所支持的最大grid的大小，这种方式仍然能够正确运行。此外，我们可以限制线程块的数量来调节性能。通常来说，线程块的数量可以设为设备所支持的多处理器的整数倍，比如：
```C++
int num_sms;
cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, devId);
// Perform add on 1M elements
add<<<32 * num_sms, 256>>>(1 << 20, x, y);
```

2. 易于调试(Debugging). 只要把线程块的数量和线程的数量都设为1，就可得到序列化处理的结果，比如：
```C++
add<<<1, 1>>>(1 << 20, x, y);
```

3. 可移植性和可读性(Portability and readability). 我们可以很容易地编写既能在GPU上运行的CUDA代码，又能在CPU上运行的代码。[Hemi](https://devblogs.nvidia.com/simple-portable-parallel-c-hemi-2) 库提供了 ```grid_stride_range()``` 来使用C++ 11的```range-based for loops```.
```C++
HEMI_LAUNCHABLE
void saxpy(int n, float a, float *x, float *y)
{
    for (auto i : hemi::grid_stride_range(0, n)) {
        y[i] = a * x[i] + y[i];
    }
}
```
调用的时候： ```hemi::cudaLaunch(saxpy, 1 << 20, x, y);```

将```add_block.cu```中的代码改写成使用 ```grid-stride loop``` 的代码如下：
```C++
// add_grid.cu

#include <iostream>
#include <cmath>


// Host: CPU, device: GPU
// CUDA Kernel function to add the elements of two arrays on the GPU
__global__
void add(int n, float* x, float* y)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
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
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    add <<<numBlocks, blockSize>>> (N, x, y);

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
```

## 5. 统计核函数的运行时间
最简单的方法是使用```nvprof```命令，当安装了```CUDA```工具箱之后就有了这个。使用```nvprof ./add_cuda```命令，可以得到核函数的运行时间。

## 6. 总结
- ```cudaMallocManaged()```
- ```cudaFree()```
- ```nvprof```
- ```threadIdx.x```
- ```blockIdx.x```
- ```blockDim.x```
- ```gridDim.x```
- ```grid-stride loop```


## Learning Bonus - Unified Memory in CUDA <sup>[2]</sup>
在CUDA 6之前，CPU和GPU之间的共享数据必须在内存上进行分配，然后需要在代码中进行显式的拷贝操作 - 这对于CUDA变成来说，增添了很多复杂性。CUDA 6之后，通过 Unified Memory, Nvidia引入了CUDA历史上最激动人心的编程模型的提升。

Unified Memory在CPU和GPU之间提供了一个内存管理池(Pool of Managed Memory)，managed memory通过一个指针就可以在CPU和GPU上进行访问。实现这个的关键在于，系统自动地在host和device之间对分配在Unified Memory上的数据进行迁移，就好像在CPU和GPU上都编写了内存访问的代码一样。如下图所示：

<dev align=center>![cuda](https://devblogs.nvidia.com/wp-content/uploads/2013/11/unified_memory.png)</dev>

先来比较两段代码
```C++
// without Unified Memory

float *x, *y;
x = (float *)malloc(N * sizeof(float));
y = (float *)malloc(N * sizeof(float));

for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
}

float *d_x, *d_y;
cudaMalloc(&d_x, N * sizeof(float));
cudaMalloc(&d_y, N * sizeof(float));

cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyDeviceToHost);
```

```C++
// with Unified Memory
float *d_x, *d_y;
cudaMallocManaged(&d_x, N * sizeof(float));
cudaMallocManaged(&d_y, N * sizeof(float));

for (int i = 0; i < N; i++) {
    d_x[i] = 1.0f;
    d_y[i] = 2.0f;
}

...

cudaDeviceSynchronize();
```
很显然，使用Unified Memory的代码更简单。只要分配一次内存，数据就均可在host和device上进行访问。

Unified Memory可以根据CPU和GPU之间的需求进行数据迁移，病同事GPU上的local data的性能和globally shared data的使用的简洁性。但是，需要记住的一点是：```a carefully tuned CUDA program that uses streams and cudaMemcpyAsync to efficiently overlap execution with data transfers may very well perform better than a CUDA program that only uses Unified Memory```！原因很简单：对于何时、何地需要数据，CUDA runtime不可能拥有比编程人员更多的信息。

接下来我们看看，Unified Memory可以做什么。

**# 消除深拷贝(Deep Copy) #**
Unified Memory的一个主要的有点就是：可以通过访问GPU kernels上的结构化数据时消除深拷贝的需要，从而对不同的内存计算模型进行简化。接下来看一些示例代码：
```C++
struct dataElem {
    int prop1;
    int prop2;
    char *name;
};
```
为了能够在device上使用这个数据结构，我们必须复制这个结构以及它的成员，并且拷贝这个结构指向的所有数据，代码如下：
```C++
void launch(dataElem *elem)
{
    dataElem *d_elem;
    char *d_name;

    int namelen = strlen(elem->name) + 1;

    // Allocate storage for struct and name
    cudaMalloc(&d_elem, sizeof(dataElem));
    cudaMalloc(&d_name, namelen);

    // Copy up each piece separately, including new "name" pointer value
    cudaMemcpy(d_elem, elem, sizeof(dataElem), cudaMemcpyHostToDevice);
    cudaMemcpy(d_name, elem->name, namelen, cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_elem->name), &d_name, sizeof(char*), cudaMemcpyHostToDevice);

    // Finally we can launch our kernel, but CPU & GPU use different copies of "elem"
    Kernel<<<...>>>(d_elem);
}
```
可以看到，在不使用Unified Memory的情况下，我们需要做大量的工作。接下来，通过一个示例，来看看通过Unified Memory，我们可以怎样简化操作。

```C++
class Managed
{
public:
    void *operator new(size_t len)
    {
        void *ptr;
        cudaMallocManaged(&ptr, len);
        cudaDeviceSynchronize();
        return ptr;
    }

    void operator delete(void *ptr) 
    {
        cudaDeviceSynchronize();
        cudaFree(ptr);
    }
};

// Deriving from "Managed" allows pass-by-reference
class String : public Managed
{
private:
    int length;
    char *data;

public:
    // Unified Memory copy constructor allows pass-by-value
    String(const String &s) 
    {
        length = s.length;
        cudaMallocManaged(&daa, length);
        memcpy(data, s.data, length);
    }
};

// Note "managed" on this class, too.
// C++ now handles our deep copies
class dataElem : public Managed
{
public:
    int prop1;
    int prop2;
    String name;
};
```
在这样做之后，我们可以在Unified Memory中分配dataElem: ```dataElem *data = new dataElem;```
## References
1. https://devblogs.nvidia.com/even-easier-introduction-cuda
2. https://devblogs.nvidia.com/unified-memory-in-cuda-6