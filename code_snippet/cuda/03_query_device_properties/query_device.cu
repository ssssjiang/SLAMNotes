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