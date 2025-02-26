#ifndef COLLIDE_SOFT_BODY_VERTICES_KERNEL_H
#define COLLIDE_SOFT_BODY_VERTICES_KERNEL_H

#include <cuda_runtime.h>  // For definition of float3

#ifdef __cplusplus
extern "C" {
#endif

// Launch the collision kernel. All GPU memory allocation is done within the .cu file.
void LaunchCollideSoftBodyVerticesKernelCUDA(
const float3* hPositions,       // Input: host vertex positions (array of float3)
const float* hInvMass,          // Input: host vertex inverse mass (array of float)
float3* hOutPositions,          // Output: vertex positions (array of float3)
float3* hCollisionPlane,        // Output: collision plane normals (array of float3)
float* hLargestPenetration,     // Output: penetration values (array of float)
int* hCollidingShapeIndex,      // Output: colliding shape index (array of int)
int numVertices,
const float hMat[16],           // Input: transformation matrix (row–major 4x4; array of 16 floats)
const float hHalfExtent[3],     // Input: half extent of bounding box (array of 3 floats)
int collidingShapeIndex);

#ifdef __cplusplus
}
#endif
#endif // COLLIDE_SOFT_BODY_VERTICES_KERNEL_H