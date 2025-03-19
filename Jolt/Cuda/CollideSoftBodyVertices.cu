__global__ void CollideSoftBodyVerticesKernelCUDA(
    const float3 *__restrict__ d_inPositions,
    const float *__restrict__ d_inInvMass,
    float3 *__restrict__ d_outPositions,
    float3 *__restrict__ d_collisionPlane,
    float *__restrict__ d_largestPenetration,
    int *__restrict__ d_collidingShapeIndex,
    int numVertices,
    CudaMat44 inverse_transform,
    float3 half_extent,
    int collidingShapeIndex)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numVertices) return;

    float3 position = d_inPositions[i];
    float invMass = d_inInvMass[i];

    // Simply copy position by default
    d_outPositions[i] = position;

    if (invMass > 0.f)
    {
        // Transform to local space:
        float3 local_pos = MultiplyMat44Float3(inverse_transform, position);

        // Clamp local_pos:
        float3 neg_half_extent = -half_extent;
        float3 clamped_point = fminf3(fmaxf3(local_pos, neg_half_extent), half_extent);

        bool inside = (local_pos.x == clamped_point.x) &&
                      (local_pos.y == clamped_point.y) &&
                      (local_pos.z == clamped_point.z);

        if (inside)
        {
            float3 delta = make_float3(half_extent.x - fabsf(local_pos.x),
                                       half_extent.y - fabsf(local_pos.y),
                                       half_extent.z - fabsf(local_pos.z));

            // Find the axis with the smallest penetration
            int axis = (delta.y < delta.x) ? ((delta.z < delta.y) ? 2 : 1) : ((delta.z < delta.x) ? 2 : 0);
            float penetration = delta[axis];

            if (d_largestPenetration[i] < penetration)
            {
                d_largestPenetration[i] = penetration;
                
                float sign = (local_pos[axis] < 0.f) ? -1.f : 1.f;
                float3 normal = make_float3(0.f, 0.f, 0.f);
                normal[axis] = sign;

                d_outPositions[i] = normal * half_extent;
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
                d_collisionPlane[i] = (norm_length > 0.f) ? diff / norm_length : make_float3(0.f, 0.f, 0.f);
                d_collidingShapeIndex[i] = collidingShapeIndex;
                d_outPositions[i] = clamped_point;
            }
        }
    }
}

// Optimized Kernel Launch Function
void LaunchCollideSoftBodyVerticesKernelCUDA(
    const float3 *hPositions,
    const float *hInvMass,
    float3 *hOutPositions,
    float3 *hCollisionPlane,
    float *hLargestPenetration,
    int *h_collidingShapeIndex,
    int numVertices,
    const float hMat[16],
    const float hHalfExtent[3],
    int collidingShapeIndex)
{
    float3 *d_inPositions = nullptr, *d_outPositions = nullptr, *d_collisionPlane = nullptr;
    float *d_inInvMass = nullptr, *d_largestPenetration = nullptr;
    int *d_collidingShapeIndex = nullptr;

    size_t posSize = numVertices * sizeof(float3);
    size_t floatSize = numVertices * sizeof(float);
    size_t intSize = numVertices * sizeof(int);

    cudaMalloc(&d_inPositions, posSize);
    cudaMalloc(&d_inInvMass, floatSize);
    cudaMalloc(&d_outPositions, posSize);
    cudaMalloc(&d_collisionPlane, posSize);
    cudaMalloc(&d_largestPenetration, floatSize);
    cudaMalloc(&d_collidingShapeIndex, intSize);

    cudaMemcpy(d_inPositions, hPositions, posSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_inInvMass, hInvMass, floatSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_largestPenetration, hLargestPenetration, floatSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_collidingShapeIndex, h_collidingShapeIndex, intSize, cudaMemcpyHostToDevice);

    CudaMat44 d_inverse;
    for (int i = 0; i < 16; ++i)
        d_inverse.m[i] = hMat[i];

    float3 d_halfExtent = make_float3(hHalfExtent[0], hHalfExtent[1], hHalfExtent[2]);

    // **Optimized Block Size**
    int blockSize = 128;  // Tuned for best occupancy
    int numBlocks = (numVertices + blockSize - 1) / blockSize;

    CollideSoftBodyVerticesKernelCUDA<<<numBlocks, blockSize>>>(
        d_inPositions, d_inInvMass, d_outPositions, d_collisionPlane,
        d_largestPenetration, d_collidingShapeIndex, numVertices, d_inverse, d_halfExtent, collidingShapeIndex);
    
    cudaDeviceSynchronize();

    cudaMemcpy(hOutPositions, d_outPositions, posSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(hCollisionPlane, d_collisionPlane, posSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(hLargestPenetration, d_largestPenetration, floatSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_collidingShapeIndex, d_collidingShapeIndex, intSize, cudaMemcpyDeviceToHost);

    cudaFree(d_inPositions);
    cudaFree(d_inInvMass);
    cudaFree(d_outPositions);
    cudaFree(d_collisionPlane);
    cudaFree(d_largestPenetration);
    cudaFree(d_collidingShapeIndex);
}