// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2021 Jorrit Rouwe
// SPDX-License-Identifier: MIT

#include <Jolt/Jolt.h>
#include <Jolt/Cuda/Hello.h>
#include <Jolt/Cuda/CollideSoftBodyVertices.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/ScaleHelpers.h>
#include <Jolt/Physics/Collision/Shape/GetTrianglesContext.h>
#include <Jolt/Physics/Collision/RayCast.h>
#include <Jolt/Physics/Collision/CastResult.h>
#include <Jolt/Physics/Collision/CollidePointResult.h>
#include <Jolt/Physics/Collision/TransformedShape.h>
#include <Jolt/Physics/Collision/CollideSoftBodyVertexIterator.h>
#include <Jolt/Geometry/RayAABox.h>
#include <Jolt/ObjectStream/TypeDeclarations.h>
#include <Jolt/Core/StreamIn.h>
#include <Jolt/Core/StreamOut.h>
#ifdef JPH_DEBUG_RENDERER
	#include <Jolt/Renderer/DebugRenderer.h>
#endif // JPH_DEBUG_RENDERER

JPH_NAMESPACE_BEGIN

JPH_IMPLEMENT_SERIALIZABLE_VIRTUAL(BoxShapeSettings)
{
	JPH_ADD_BASE_CLASS(BoxShapeSettings, ConvexShapeSettings)

	JPH_ADD_ATTRIBUTE(BoxShapeSettings, mHalfExtent)
	JPH_ADD_ATTRIBUTE(BoxShapeSettings, mConvexRadius)
}

static const Vec3 sUnitBoxTriangles[] = {
	Vec3(-1, 1, -1),	Vec3(-1, 1, 1),		Vec3(1, 1, 1),
	Vec3(-1, 1, -1),	Vec3(1, 1, 1),		Vec3(1, 1, -1),
	Vec3(-1, -1, -1),	Vec3(1, -1, -1),	Vec3(1, -1, 1),
	Vec3(-1, -1, -1),	Vec3(1, -1, 1),		Vec3(-1, -1, 1),
	Vec3(-1, 1, -1),	Vec3(-1, -1, -1),	Vec3(-1, -1, 1),
	Vec3(-1, 1, -1),	Vec3(-1, -1, 1),	Vec3(-1, 1, 1),
	Vec3(1, 1, 1),		Vec3(1, -1, 1),		Vec3(1, -1, -1),
	Vec3(1, 1, 1),		Vec3(1, -1, -1),	Vec3(1, 1, -1),
	Vec3(-1, 1, 1),		Vec3(-1, -1, 1),	Vec3(1, -1, 1),
	Vec3(-1, 1, 1),		Vec3(1, -1, 1),		Vec3(1, 1, 1),
	Vec3(-1, 1, -1),	Vec3(1, 1, -1),		Vec3(1, -1, -1),
	Vec3(-1, 1, -1),	Vec3(1, -1, -1),	Vec3(-1, -1, -1)
};

ShapeSettings::ShapeResult BoxShapeSettings::Create() const
{
	if (mCachedResult.IsEmpty())
		Ref<Shape> shape = new BoxShape(*this, mCachedResult);
	return mCachedResult;
}

BoxShape::BoxShape(const BoxShapeSettings &inSettings, ShapeResult &outResult) :
	ConvexShape(EShapeSubType::Box, inSettings, outResult),
	mHalfExtent(inSettings.mHalfExtent),
	mConvexRadius(inSettings.mConvexRadius)
{
	// Check convex radius
	if (inSettings.mConvexRadius < 0.0f
		|| inSettings.mHalfExtent.ReduceMin() <= inSettings.mConvexRadius)
	{
		outResult.SetError("Invalid convex radius");
		return;
	}

	// Result is valid
	outResult.Set(this);
}

class BoxShape::Box final : public Support
{
public:
					Box(const AABox &inBox, float inConvexRadius) :
		mBox(inBox),
		mConvexRadius(inConvexRadius)
	{
		static_assert(sizeof(Box) <= sizeof(SupportBuffer), "Buffer size too small");
		JPH_ASSERT(IsAligned(this, alignof(Box)));
	}

	virtual Vec3	GetSupport(Vec3Arg inDirection) const override
	{
		return mBox.GetSupport(inDirection);
	}

	virtual float	GetConvexRadius() const override
	{
		return mConvexRadius;
	}

private:
	AABox			mBox;
	float			mConvexRadius;
};

const ConvexShape::Support *BoxShape::GetSupportFunction(ESupportMode inMode, SupportBuffer &inBuffer, Vec3Arg inScale) const
{
	// Scale our half extents
	Vec3 scaled_half_extent = inScale.Abs() * mHalfExtent;

	switch (inMode)
	{
	case ESupportMode::IncludeConvexRadius:
	case ESupportMode::Default:
		{
			// Make box out of our half extents
			AABox box = AABox(-scaled_half_extent, scaled_half_extent);
			JPH_ASSERT(box.IsValid());
			return new (&inBuffer) Box(box, 0.0f);
		}

	case ESupportMode::ExcludeConvexRadius:
		{
			// Reduce the box by our convex radius
			float convex_radius = ScaleHelpers::ScaleConvexRadius(mConvexRadius, inScale);
			Vec3 convex_radius3 = Vec3::sReplicate(convex_radius);
			Vec3 reduced_half_extent = scaled_half_extent - convex_radius3;
			AABox box = AABox(-reduced_half_extent, reduced_half_extent);
			JPH_ASSERT(box.IsValid());
			return new (&inBuffer) Box(box, convex_radius);
		}
	}

	JPH_ASSERT(false);
	return nullptr;
}

void BoxShape::GetSupportingFace(const SubShapeID &inSubShapeID, Vec3Arg inDirection, Vec3Arg inScale, Mat44Arg inCenterOfMassTransform, SupportingFace &outVertices) const
{
	JPH_ASSERT(inSubShapeID.IsEmpty(), "Invalid subshape ID");

	Vec3 scaled_half_extent = inScale.Abs() * mHalfExtent;
	AABox box(-scaled_half_extent, scaled_half_extent);
	box.GetSupportingFace(inDirection, outVertices);

	// Transform to world space
	for (Vec3 &v : outVertices)
		v = inCenterOfMassTransform * v;
}

MassProperties BoxShape::GetMassProperties() const
{
	MassProperties p;
	p.SetMassAndInertiaOfSolidBox(2.0f * mHalfExtent, GetDensity());
	return p;
}

Vec3 BoxShape::GetSurfaceNormal(const SubShapeID &inSubShapeID, Vec3Arg inLocalSurfacePosition) const
{
	JPH_ASSERT(inSubShapeID.IsEmpty(), "Invalid subshape ID");

	// Get component that is closest to the surface of the box
	int index = (inLocalSurfacePosition.Abs() - mHalfExtent).Abs().GetLowestComponentIndex();

	// Calculate normal
	Vec3 normal = Vec3::sZero();
	normal.SetComponent(index, inLocalSurfacePosition[index] > 0.0f? 1.0f : -1.0f);
	return normal;
}

#ifdef JPH_DEBUG_RENDERER
void BoxShape::Draw(DebugRenderer *inRenderer, RMat44Arg inCenterOfMassTransform, Vec3Arg inScale, ColorArg inColor, bool inUseMaterialColors, bool inDrawWireframe) const
{
	DebugRenderer::EDrawMode draw_mode = inDrawWireframe? DebugRenderer::EDrawMode::Wireframe : DebugRenderer::EDrawMode::Solid;
	inRenderer->DrawBox(inCenterOfMassTransform * Mat44::sScale(inScale.Abs()), GetLocalBounds(), inUseMaterialColors? GetMaterial()->GetDebugColor() : inColor, DebugRenderer::ECastShadow::On, draw_mode);
}
#endif // JPH_DEBUG_RENDERER

bool BoxShape::CastRay(const RayCast &inRay, const SubShapeIDCreator &inSubShapeIDCreator, RayCastResult &ioHit) const
{
	// Test hit against box
	float fraction = max(RayAABox(inRay.mOrigin, RayInvDirection(inRay.mDirection), -mHalfExtent, mHalfExtent), 0.0f);
	if (fraction < ioHit.mFraction)
	{
		ioHit.mFraction = fraction;
		ioHit.mSubShapeID2 = inSubShapeIDCreator.GetID();
		return true;
	}
	return false;
}

void BoxShape::CastRay(const RayCast &inRay, const RayCastSettings &inRayCastSettings, const SubShapeIDCreator &inSubShapeIDCreator, CastRayCollector &ioCollector, const ShapeFilter &inShapeFilter) const
{
	// Test shape filter
	if (!inShapeFilter.ShouldCollide(this, inSubShapeIDCreator.GetID()))
		return;

	float min_fraction, max_fraction;
	RayAABox(inRay.mOrigin, RayInvDirection(inRay.mDirection), -mHalfExtent, mHalfExtent, min_fraction, max_fraction);
	if (min_fraction <= max_fraction // Ray should intersect
		&& max_fraction >= 0.0f // End of ray should be inside box
		&& min_fraction < ioCollector.GetEarlyOutFraction()) // Start of ray should be before early out fraction
	{
		// Better hit than the current hit
		RayCastResult hit;
		hit.mBodyID = TransformedShape::sGetBodyID(ioCollector.GetContext());
		hit.mSubShapeID2 = inSubShapeIDCreator.GetID();

		// Check front side
		if (inRayCastSettings.mTreatConvexAsSolid || min_fraction > 0.0f)
		{
			hit.mFraction = max(0.0f, min_fraction);
			ioCollector.AddHit(hit);
		}

		// Check back side hit
		if (inRayCastSettings.mBackFaceModeConvex == EBackFaceMode::CollideWithBackFaces
			&& max_fraction < ioCollector.GetEarlyOutFraction())
		{
			hit.mFraction = max_fraction;
			ioCollector.AddHit(hit);
		}
	}
}

void BoxShape::CollidePoint(Vec3Arg inPoint, const SubShapeIDCreator &inSubShapeIDCreator, CollidePointCollector &ioCollector, const ShapeFilter &inShapeFilter) const
{
	// Test shape filter
	if (!inShapeFilter.ShouldCollide(this, inSubShapeIDCreator.GetID()))
		return;

	if (Vec3::sLessOrEqual(inPoint.Abs(), mHalfExtent).TestAllXYZTrue())
		ioCollector.AddHit({ TransformedShape::sGetBodyID(ioCollector.GetContext()), inSubShapeIDCreator.GetID() });
}

void PrintMat44(const JPH::Mat44 &mat) {
    // JPH::Mat44 is stored column-major, so we extract by columns
    for (int row = 0; row < 4; row++) {
        for (int col = 0; col < 4; col++) {
            printf("%f ", mat(row, col));  // Direct access using operator()(row, col)
        }
        printf("\n");  // Newline for formatting
    }
}

int MatrixCounter = 0;
void BoxShape::CollideSoftBodyVertices(Mat44Arg inCenterOfMassTransform, 
                                         Vec3Arg inScale, 
                                         const CollideSoftBodyVertexIterator &inVertices, 
                                         uint inNumVertices, 
                                         int inCollidingShapeIndex) const
{
    printf("Launch Kernel\n");
    
    // Compute half extents.
    Vec3 half_extent = inScale.Abs() * mHalfExtent;
    printf("Half extent computed: (%f, %f, %f)\n", half_extent.GetX(), half_extent.GetY(), half_extent.GetZ());
    
    // Print out a few vertices for verification.
    if (inNumVertices == 0)
    {
        printf("No vertices!\n");
        return;
    }

	for (CollideSoftBodyVertexIterator v = inVertices, end = inVertices + inNumVertices; v != end; ++v)
	{
		Vec3 pos = v.GetPosition();
		printf("Vertex position: (%f, %f, %f)\n", pos.GetX(), pos.GetY(), pos.GetZ());
	}
    
	for (uint i = 0; i < std::min((unsigned)5, inNumVertices); ++i)
	{
	    const auto &vertex = inVertices[i];
	    Vec3 pos = vertex.GetPosition();
	    printf("Vertex %u: pos=(%f, %f, %f)\n", i, pos.GetX(), pos.GetY(), pos.GetZ());
	}
    
    // Prepare host arrays for vertex data.
    std::vector<float3> positions(inNumVertices);
    std::vector<float> invMass(inNumVertices);
    std::vector<float3> outPositions(inNumVertices);
    std::vector<float3> collisionPlane(inNumVertices);
    std::vector<float> largestPenetration(inNumVertices, -1e9f);
    std::vector<int> collidingShapeIndexArr(inNumVertices, -1);
    
    // Fill arrays from the soft body vertex iterator.
    //for (uint i = 0; i < inNumVertices; ++i)
    //{
    //    const auto &vertex = inVertices[i];
    //    Vec3 pos = vertex.GetPosition();
    //    positions[i] = make_float3(pos.GetX(), pos.GetY(), pos.GetZ());
    //    invMass[i] = vertex.GetInvMass();
    //}
	int i = 0;
	for (CollideSoftBodyVertexIterator v = inVertices, end = inVertices + inNumVertices; v != end; ++v, ++i)
	{
		Vec3 pos = v.GetPosition();
		positions[i] = make_float3(pos.GetX(), pos.GetY(), pos.GetZ());
		invMass[i] = v.GetInvMass();
	}
    
    // Convert transformation matrix.
    Mat44 mat = inCenterOfMassTransform.ToMat44();
    float hMat[16];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
        {
            hMat[i * 4 + j] = mat(i, j);
            // Optionally print each element:
            // printf("hMat[%d] = %f\n", i * 4 + j, hMat[i * 4 + j]);
        }
    
    float hHalfExtent[3] = { half_extent.GetX(), half_extent.GetY(), half_extent.GetZ() };
    printf("Matrix and half extent prepared\n");
    
    // Call CUDA kernel launcher.
    LaunchCollideSoftBodyVerticesKernelCUDA(
         positions.data(),
         invMass.data(),
         outPositions.data(),
         collisionPlane.data(),
         largestPenetration.data(),
         collidingShapeIndexArr.data(),
         inNumVertices,
         hMat,
         hHalfExtent,
         inCollidingShapeIndex
    );
    
    printf("Kernel Launched\n");
    
    // Update vertex collision data based on results from the kernel.
	// use the same CollideSoftBodyVertexIterator to update the vertices
	uint idx = 0;
    // Create non-const copy if you need to modify the vertices.
    CollideSoftBodyVertexIterator v = inVertices;
    CollideSoftBodyVertexIterator end = inVertices + inNumVertices;
    for (; v != end; ++v, ++idx)
    {
        if (collidingShapeIndexArr[idx] != -1)
        {
             Vec3 normal(collisionPlane[idx].x, collisionPlane[idx].y, collisionPlane[idx].z);
             Vec3 point = normal * half_extent;  // contact point (example)
             Plane collision = Plane::sFromPointAndNormal(point, normal);
             collision = collision.GetTransformed(inCenterOfMassTransform);
             // Use the iterator to update the vertex collision.
             v.SetCollision(collision, collidingShapeIndexArr[idx]);
        }
    }
	printf("Vertex collision data updated\n");
}
//void BoxShape::CollideSoftBodyVertices(Mat44Arg inCenterOfMassTransform, Vec3Arg inScale, const CollideSoftBodyVertexIterator &inVertices, uint inNumVertices, int inCollidingShapeIndex) const
//{
//	// Print the inCenterOfMassTransform and what 
//	//printf("MatrixCounter: %d\n", MatrixCounter);
//	//Mat44 mat = inCenterOfMassTransform.ToMat44();
//
//	//printf("Mat44Arg has size:" + sizeof(typeof(Mat44)));
//	//printf("Vec3Arg has size:" + sizeof(typeof(Vec3)));
//	//printf("Vec3Arg has size:" + sizeof(typeof(Collide SoftBodyVertexIterator))); // per soft body
//	//printf("Vec3Arg has size:" + sizeof(typeof(uint)));
//	//printf("Vec3Arg has size:" + sizeof(typeof(int)));
//	//sizeof(typeof(Mat44));
//	printf("Launch Kernel\n");
//    LaunchHelloKernel();   // Invoke CUDA kernel
//	printf("Kernel Launched\n");
//
//
//	//printf("Matrix contents:\n");
//	//PrintMat44(mat);  // Print the matrix
//	MatrixCounter++;
//	Mat44 inverse_transform = inCenterOfMassTransform.InversedRotationTranslation();
//	Vec3 half_extent = inScale.Abs() * mHalfExtent;
//
//	for (CollideSoftBodyVertexIterator v = inVertices, sbv_end = inVertices + inNumVertices; v != sbv_end; ++v)
//		if (v.GetInvMass() > 0.0f)
//		{
//			// Convert to local space
//			Vec3 local_pos = inverse_transform * v.GetPosition();
//
//			// Clamp point to inside box
//			Vec3 clamped_point = Vec3::sMax(Vec3::sMin(local_pos, half_extent), -half_extent);
//
//			// Test if point was inside
//			if (clamped_point == local_pos)
//			{
//				// Calculate closest distance to surface
//				Vec3 delta = half_extent - local_pos.Abs();
//				int index = delta.GetLowestComponentIndex();
//				float penetration = delta[index];
//				if (v.UpdatePenetration(penetration))
//				{
//					// Calculate contact point and normal
//					Vec3 possible_normals[] = { Vec3::sAxisX(), Vec3::sAxisY(), Vec3::sAxisZ() };
//					Vec3 normal = local_pos.GetSign() * possible_normals[index];
//					Vec3 point = normal * half_extent;
//
//					// Store collision
//					v.SetCollision(Plane::sFromPointAndNormal(point, normal).GetTransformed(inCenterOfMassTransform), inCollidingShapeIndex);
//				}
//			}
//			else
//			{
//				// Calculate normal
//				Vec3 normal = local_pos - clamped_point;
//				float normal_length = normal.Length();
//
//				// Penetration will be negative since we're not penetrating
//				float penetration = -normal_length;
//				if (v.UpdatePenetration(penetration))
//				{
//					normal /= normal_length;
//
//					// Store collision
//					v.SetCollision(Plane::sFromPointAndNormal(clamped_point, normal).GetTransformed(inCenterOfMassTransform), inCollidingShapeIndex);
//				}
//			}
//		}
//}
//
void BoxShape::GetTrianglesStart(GetTrianglesContext &ioContext, const AABox &inBox, Vec3Arg inPositionCOM, QuatArg inRotation, Vec3Arg inScale) const
{
	new (&ioContext) GetTrianglesContextVertexList(inPositionCOM, inRotation, inScale, Mat44::sScale(mHalfExtent), sUnitBoxTriangles, std::size(sUnitBoxTriangles), GetMaterial());
}

int BoxShape::GetTrianglesNext(GetTrianglesContext &ioContext, int inMaxTrianglesRequested, Float3 *outTriangleVertices, const PhysicsMaterial **outMaterials) const
{
	return ((GetTrianglesContextVertexList &)ioContext).GetTrianglesNext(inMaxTrianglesRequested, outTriangleVertices, outMaterials);
}

void BoxShape::SaveBinaryState(StreamOut &inStream) const
{
	ConvexShape::SaveBinaryState(inStream);

	inStream.Write(mHalfExtent);
	inStream.Write(mConvexRadius);
}

void BoxShape::RestoreBinaryState(StreamIn &inStream)
{
	ConvexShape::RestoreBinaryState(inStream);

	inStream.Read(mHalfExtent);
	inStream.Read(mConvexRadius);
}

void BoxShape::sRegister()
{
	ShapeFunctions &f = ShapeFunctions::sGet(EShapeSubType::Box);
	f.mConstruct = []() -> Shape * { return new BoxShape; };
	f.mColor = Color::sGreen;
}

JPH_NAMESPACE_END
