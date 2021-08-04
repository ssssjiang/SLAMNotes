# 04. Separate Compilation and Linking of CUDA C++ Device Code

之前的代码都是写在一个CUDA文件中进行编译。现在，我们需要对多个CUDA代码进行编译和链接。

## 1. 示例代码

```C++
// v3.cuh
class v3
{
public:
    float _x;
    float _y;
    float _z;

    v3();
    v3(float x, float y, float z);
    void randomize();
    __host__ __device__ void normalize();
    __host__ __device__ void scramble();
};
```

```C++
// v3.cu
#include "v3.cuh"

#include <cmath>
#include <random>

v3::v3() { randomize(); }

v3::v3(float x, float y, float z) : _x(x), _y(y), _z(z) { }

void v3::randomize()
{
    _x = (float)rand() / (float)RAND_MAX;
    _y = (float)rand() / (float)RAND_MAX;
    _z = (float)rand() / (float)RAND_MAX;
}   

__host__ __device__ void v3::normalize()
{
    float t = sqrt(_x * _x + _y * _y + _z * _z);
    _x /= t;
    _y /= t;
    _z /= t;
}

__host__ __device__ void v3::scramble()
{
    float tx = 0.317f * (_x + 1.0) + _y + _z * _x * _x + _y + _z;
    float ty = 0.619f * (_y + 1.0) + _y * _y + _x * _y * _z + _y + _x;
    float tz = 0.124f * (_z + 1.0) + _z * _y + _x * _y * _z + _y + _x;
    _x = tx;
    _y = ty;
    _z = tz;
}
```

```C++
// particle.cuh

#include "v3.cuh"

class particle
{
private:
    v3 position;
    v3 velocity;
    v3 totalDistance;
public:
    particle();
    __host__ __device__ void advance(float dist);
    const v3& getTotalDistance() const;
};
```

```C++
// particle.cu

#include "particle.cuh"

particle::particle() : position(), velocity(), totalDistance(0, 0, 0) {}

__host__ __device__ void particle::advance(float d)
{
    velocity.normalize();
    float dx = d * velocity._x;
    position._x += dx;
    totalDistance._x += dx;
    float dy = d * velocity._y;
    position._y += dy;
    totalDistance._y += dy;
    float dz = d * velocity._z;
    position._z += dz;
    totalDistance._z += dz;
    velocity.scramble();
}

const v3& particle::getTotalDistance() const
{
    return totalDistance;
}
```

```C++
#include "particle.cuh"

#include <cstdlib>
#include <cstdio>

__global__
void advanceParticles(float dt, particle *pArray, int nParticles)
{
    int idx = threadIdx.x + blockIdx.x * blockIdx.x * blockDim.x;
    if (idx < nParticles) { pArray[idx].advance(dt); }
}

int main(int argc, char **argv)
{
    int n = 1000000;
    if (argc > 1) { n = atoi(argv[1]); } // number of particles
    if (argc > 2) { srand(atoi(argv[2])); } // random seed
    
    particle *pArray = new particle[n];
    particle *devPArray = NULL;
    cudaMalloc(&devPArray, n * sizeof(particle));
    cudaMemcpy(devPArray, pArray, n * sizeof(particle), cudaMemcpyHostToDevice);
    for (int i = 0; i < 100; i++) {
        // Random distance each step
        float dt = (float)rand() / (float) RAND_MAX;
        advanceParticles<<< 1 + n / 256, 256>>>(dt, devPArray, n);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(pArray, devPArray, n * sizeof(particle), cudaMemcpyDeviceToHost);
    v3 totalDistance(0, 0, 0);
    v3 temp;
    for (int i = 0; i < n; i++) {
        temp = pArray[i].getTotalDistance();
        totalDistance._x += temp._x;
        totalDistance._y += temp._y;
        totalDistance._z += temp._z;
    }

    float avgX = totalDistance._x / (float)n;
    float avgY = totalDistance._y / (float)n;
    float avgZ = totalDistance._z / (float)n;
    float avgNorm = sqrt(avgX * avgX + avgY * avgY + avgZ * avgZ);
    printf("Moved %d particles 100 steps. Average distance traveled is |(%f, %f, %f)| = %f\n",
           n, avgX, avgY, avgZ, avgNorm);
}
```
其中需要注意的地方是，```v3.cuh```，```v3.cu```, ```particle.cuh```, ```particle.cu```中的```__host__ __device```声明。这两个声明告诉```NVCC```生成在host和device上均可访问的代码。当然也可以单独声明其中一个，默认情况下是```__host__```。


## 使用CMake编译

在参考的博客中<sup>[1]</sup>，作者使用```Makefile```对CUDA代码进行```separate compilation```。
```Makefile
objects = main.o particle.o v3.o

all: $(objects)
    nvcc -arch=sm_20 $(objects) -o app

%.o: %.cpp
    nvcc -x cu -arch=sm_20 -I. -dc $< -o $@

clean:
    rm -rf *.o.app
```
需要注意的地方是```-arch=sm_20```. 这个选项是必须的，因为并不是所有的```SM```代码变种都支持设备链接，并且```NVCC```需要知道它的目标与SM架构是适配的(这里大致意思如此，翻译过来可能不准确，具体参考[1])。

但是，现在的工程项目基本都使用```CMake```。因此，我们需要知道如何在```CMakeLists.txt```中配置```CUDA```。

首先：
```cmake
set(CUDA_NVCC_FLAGS -arch=sm_30
                    -gencode=arch=compute_30,code=sm_35
                    -gencode=arch=compute_35,code=sm_35
                    -gencode=arch=compute_50,code=sm_50
                    -gencode=arch=compute_52,code=sm_52
                    -gencode=arch=compute_60,code=sm_60
                    -gencode=arch=compute_61,code=sm_61
                    -gencode=arch=compute_61,code=compute_61
                    )
```
关于这么配置的原因，参考[4]中介绍。

接着需要配置```NVCC```的编译选项，我们需要加入
```
set(CUDA_SEPARABLE_COMPILATION ON)
```
如果没有这句，会出现```ptxas fatal   : Unresolved extern function '_ZN8particle7advanceEf'```。出现这个问题的原因是
```The issue is that you defined a __device__ function in separate compilation unit from __global__ that calls it. You need to either explicitely enable relocatable device code mode by adding -dc flag or move your definition to the same unit.```<sup>[3]</sup>这句话正确地指出了原因，但是解决方法是错的。应该修正为我们这里使用的```CMake```语句。

然后，查找CUDA包：```find_package(CUDA REQUIRED)```

如果想生成库文件，需要使用```cuda_add_library(libname *.cuh *.cu)```; 如果生成可执行文件，需要使用```cuda_add_executable(exe_name *.cuh *.cu)```。

完整CMakeLists.txt如下：
```CMake
cmake_minimum_required(VERSION 3.5)
project(compilation)

set(CMAKE_CXX_FLAGS "-std=c++11")
set(CUDA_NVCC_FLAGS -arch=sm_30
                    -gencode=arch=compute_30,code=sm_35
                    -gencode=arch=compute_35,code=sm_35
                    -gencode=arch=compute_50,code=sm_50
                    -gencode=arch=compute_52,code=sm_52
                    -gencode=arch=compute_60,code=sm_60
                    -gencode=arch=compute_61,code=sm_61
                    -gencode=arch=compute_61,code=compute_61
                    )

# set(CUDA_NVCC_FLAGS --relocatable-device-code=true)
set(CUDA_SEPARABLE_COMPILATION ON)

# get_filename_component(CUDA_LIB_PATH ${CUDA_CUDART_LIBRARY} DIRECTORY)
# find_library(CUDA_cudadevrt_LIBRARY cudadevrt PATHS ${CUDA_LIB_PATH})


find_package(CUDA QUIET REQUIRED)

# include_directories(${CUDA_TOOLKIT_INCLUDE})

cuda_add_executable(main v3.cuh particle.cuh
                         v3.cu particle.cu main.cu)
target_link_libraries(main # ${CUDA_LIBRARIES} 
                        #    ${CUDA_cudadevrt_LIBRARY}
                           )
```


## References

[1] https://devblogs.nvidia.com/separate-compilation-linking-cuda-device-code

[2] https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#nvcc-command-options

[3] https://stackoverflow.com/questions/31006581/cuda-device-unresolved-extern-function

[4] http://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
