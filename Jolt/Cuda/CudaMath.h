#ifndef CUDAMATH_H
#define CUDAMATH_H

#include <cuda_runtime.h>
#include <math.h>

// Define CUDA_FUNC for functions that need __host__ and __device__ when compiled with NVCC.
#ifdef __CUDACC__
    #define CUDA_FUNC __host__ __device__ inline
#else
    #define CUDA_FUNC inline
#endif

// Overloaded operators on float3:

CUDA_FUNC float3 operator+(const float3 &a, const float3 &b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

CUDA_FUNC float3 operator-(const float3 &a, const float3 &b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

CUDA_FUNC float3 operator*(const float3 &a, float s)
{
    return make_float3(a.x * s, a.y * s, a.z * s);
}

CUDA_FUNC float3 operator*(float s, const float3 &a)
{
    return make_float3(s * a.x, s * a.y, s * a.z);
}

CUDA_FUNC float3 operator/(const float3 &a, float s)
{
    return make_float3(a.x / s, a.y / s, a.z / s);
}

CUDA_FUNC float lengthF(const float3 &a)
{
    return sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
}

CUDA_FUNC float3 normalizeF(const float3 &a)
{
    float len = lengthF(a);
    return (len > 0.f) ? a * (1.0f / len) : a;
}

CUDA_FUNC float3 fabsf3(const float3 &a)
{
    return make_float3(fabsf(a.x), fabsf(a.y), fabsf(a.z));
}

CUDA_FUNC float3 fminf3(const float3 &a, const float3 &b)
{
    return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}

CUDA_FUNC float3 fmaxf3(const float3 &a, const float3 &b)
{
    return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}

// A simple CUDA-friendly 4x4 matrix type in row–major order.
struct CudaMat44
{
    float m[16]; // rows: m[0..3]=row0, m[4..7]=row1, etc.
};

// Multiply a 4x4 matrix by a float3 (assume vector is (x,y,z,1))
CUDA_FUNC float3 MultiplyMat44Float3(const CudaMat44 &M, const float3 &v)
{
    float3 result;
    result.x = M.m[0]  * v.x + M.m[1]  * v.y + M.m[2]  * v.z + M.m[3];
    result.y = M.m[4]  * v.x + M.m[5]  * v.y + M.m[6]  * v.z + M.m[7];
    result.z = M.m[8]  * v.x + M.m[9]  * v.y + M.m[10] * v.z + M.m[11];
    return result;
}

#endif // CUDAMATH_H