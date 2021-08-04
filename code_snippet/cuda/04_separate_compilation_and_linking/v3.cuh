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