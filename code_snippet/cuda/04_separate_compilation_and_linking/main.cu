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