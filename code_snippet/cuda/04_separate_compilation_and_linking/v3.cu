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