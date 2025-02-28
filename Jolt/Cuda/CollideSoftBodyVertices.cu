#include "CudaMath.h"
#include "CollideSoftBodyVertices.h"
#include <cuda_runtime.h>
#include <stdio.h>

// CUDA kernel using CUDA-friendly types
__global__ void CollideSoftBodyVerticesKernelCUDA(
    const float3 *d_inPositions,
    const float *d_inInvMass,
    float3 *d_outPositions,
    float3 *d_collisionPlane,
    float *d_largestPenetration,
    int *d_collidingShapeIndex,
    int numVertices,
    CudaMat44 inverse_transform,
    float3 half_extent,
    int collidingShapeIndex)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numVertices)
        return;

    float3 position = d_inPositions[i];
    float invMass = d_inInvMass[i];

    // For this example, simply copy the position
    d_outPositions[i] = position;

    if (invMass > 0.f)
    {
        // Transform to local space:
        float3 local_pos = MultiplyMat44Float3(inverse_transform, position);

        // Clamp local_pos:
        float3 neg_half_extent = make_float3(-half_extent.x, -half_extent.y, -half_extent.z);
        float3 temp = fminf3(local_pos, half_extent);
        float3 clamped_point = fmaxf3(temp, neg_half_extent);

        bool inside = (local_pos.x == clamped_point.x) &&
                      (local_pos.y == clamped_point.y) &&
                      (local_pos.z == clamped_point.z);

        if (inside)
        {
            // Compute penetration along each axis:
            float3 abs_local = fabsf3(local_pos);
            float3 delta;
            delta.x = half_extent.x - abs_local.x;
            delta.y = half_extent.y - abs_local.y;
            delta.z = half_extent.z - abs_local.z;

            int axis = 0;
            float penetration = delta.x;
            if (delta.y < penetration) { axis = 1; penetration = delta.y; }
            if (delta.z < penetration) { axis = 2; penetration = delta.z; }

            if (d_largestPenetration[i] < penetration)
            {
                d_largestPenetration[i] = penetration;
                float3 sign;
                sign.x = (local_pos.x < 0.f) ? -1.f : 1.f;
                sign.y = (local_pos.y < 0.f) ? -1.f : 1.f;
                sign.z = (local_pos.z < 0.f) ? -1.f : 1.f;

                float3 possible_normals[3] = {
                    make_float3(1.f, 0.f, 0.f),
                    make_float3(0.f, 1.f, 0.f),
                    make_float3(0.f, 0.f, 1.f)
                };

                float3 normal = make_float3(0.f, 0.f, 0.f);
                if (axis == 0)
                    normal = make_float3(sign.x * possible_normals[0].x, 0.f, 0.f);
                else if (axis == 1)
                    normal = make_float3(0.f, sign.y * possible_normals[1].y, 0.f);
                else if (axis == 2)
                    normal = make_float3(0.f, 0.f, sign.z * possible_normals[2].z);
                
                d_outPositions[i] = make_float3(half_extent.x * normal.x, half_extent.y * normal.y, half_extent.z * normal.z);
                d_collisionPlane[i] = normal;
                d_collidingShapeIndex[i] = collidingShapeIndex;
            }
        }
        else
        {
            float3 diff = local_pos - clamped_point;
            float norm_length = lengthF(diff);
            float penetration = -norm_length;
            if (d_largestPenetration[i] < penetration)
            {
                d_largestPenetration[i] = penetration;
                float3 norm = (norm_length > 0.f) ? diff / norm_length : make_float3(0.f,0.f,0.f);
                d_collisionPlane[i] = norm;
                d_collidingShapeIndex[i] = collidingShapeIndex;
                d_outPositions[i] = clamped_point;
            }
        }
    }
}

void LaunchCollideSoftBodyVerticesKernelCUDA(
    const float3* hPositions,
    const float* hInvMass,
    float3* hOutPositions,
    float3* hCollisionPlane,
    float* hLargestPenetration,
    int* h_collidingShapeIndex,
    int numVertices,
    const float hMat[16],
    const float hHalfExtent[3],
    int collidingShapeIndex)
{
    printf("Launching CollideSoftBodyVerticesKernelCUDA...\n");
    float3 *d_inPositions = nullptr, *d_outPositions = nullptr, *d_collisionPlane = nullptr;
    float  *d_inInvMass = nullptr, *d_largestPenetration = nullptr;
    int    *d_collidingShapeIndex = nullptr;

    size_t posSize = numVertices * sizeof(float3);
    size_t floatSize = numVertices * sizeof(float);
    size_t intSize = numVertices * sizeof(int);

    cudaError_t err = cudaMalloc(&d_inPositions, posSize);
    if (err != cudaSuccess) { printf("cudaMalloc d_inPositions: %s\n", cudaGetErrorString(err)); return; }
    err = cudaMalloc(&d_inInvMass, floatSize);
    if (err != cudaSuccess) { printf("cudaMalloc d_inInvMass: %s\n", cudaGetErrorString(err)); return; }
    err = cudaMalloc(&d_outPositions, posSize);
    if (err != cudaSuccess) { printf("cudaMalloc d_outPositions: %s\n", cudaGetErrorString(err)); return; }
    err = cudaMalloc(&d_collisionPlane, posSize);
    if (err != cudaSuccess) { printf("cudaMalloc d_collisionPlane: %s\n", cudaGetErrorString(err)); return; }
    err = cudaMalloc(&d_largestPenetration, floatSize);
    if (err != cudaSuccess) { printf("cudaMalloc d_largestPenetration: %s\n", cudaGetErrorString(err)); return; }
    err = cudaMalloc(&d_collidingShapeIndex, intSize);
    if (err != cudaSuccess) { printf("cudaMalloc d_collidingShapeIndex: %s\n", cudaGetErrorString(err)); return; }

    err = cudaMemcpy(d_inPositions, hPositions, posSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printf("cudaMemcpy d_inPositions: %s\n", cudaGetErrorString(err)); return; }
    err = cudaMemcpy(d_inInvMass, hInvMass, floatSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printf("cudaMemcpy d_inInvMass: %s\n", cudaGetErrorString(err)); return; }
    err = cudaMemcpy(d_largestPenetration, hLargestPenetration, floatSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printf("cudaMemcpy d_largestPenetration: %s\n", cudaGetErrorString(err)); return; }
    err = cudaMemcpy(d_collidingShapeIndex, h_collidingShapeIndex, intSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printf("cudaMemcpy d_collidingShapeIndex: %s\n", cudaGetErrorString(err)); return; }

    CudaMat44 d_inverse;
    for (int i = 0; i < 16; ++i)
        d_inverse.m[i] = hMat[i];

    float3 d_halfExtent = make_float3(hHalfExtent[0], hHalfExtent[1], hHalfExtent[2]);

    int blockSize = 256;
    int numBlocks = (numVertices + blockSize - 1) / blockSize;
    CollideSoftBodyVerticesKernelCUDA<<<numBlocks, blockSize>>>(
         d_inPositions, d_inInvMass, d_outPositions, d_collisionPlane,
         d_largestPenetration, d_collidingShapeIndex,
         numVertices, d_inverse, d_halfExtent, collidingShapeIndex);
    err = cudaGetLastError();
    if (err != cudaSuccess) 
    {
         printf("Kernel launch error: %s\n", cudaGetErrorString(err));
         return;
    }
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
         printf("Kernel after sync error: %s\n", cudaGetErrorString(err));
         return;
    }

    err = cudaMemcpy((void*)hOutPositions, d_outPositions, posSize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { printf("cudaMemcpy hOutPositions: %s\n", cudaGetErrorString(err)); return; }
    err = cudaMemcpy((void*)hCollisionPlane, d_collisionPlane, posSize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { printf("cudaMemcpy hCollisionPlane: %s\n", cudaGetErrorString(err)); return; }
    err = cudaMemcpy((void*)hLargestPenetration, d_largestPenetration, floatSize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { printf("cudaMemcpy hLargestPenetration: %s\n", cudaGetErrorString(err)); return; }
    err = cudaMemcpy((void*)h_collidingShapeIndex, d_collidingShapeIndex, intSize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { printf("cudaMemcpy d_collidingShapeIndex: %s\n", cudaGetErrorString(err)); return; }

    cudaFree(d_inPositions);
    cudaFree(d_inInvMass);
    cudaFree(d_outPositions);
    cudaFree(d_collisionPlane);
    cudaFree(d_largestPenetration);
    cudaFree(d_collidingShapeIndex);
}