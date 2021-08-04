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