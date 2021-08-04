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