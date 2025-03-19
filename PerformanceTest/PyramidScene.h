// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2023 Jorrit Rouwe
// SPDX-License-Identifier: MIT

#pragma once

// Jolt includes
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
// Our includes 
#include <Jolt/Physics/SoftBody/SoftBodyCreationSettings.h>
#include <Jolt/Physics/SoftBody/SoftBodyMotionProperties.h>
#include <Jolt/Physics/SoftBody/SoftBodySharedSettings.h>

// Local includes
#include "PerformanceTestScene.h"
#include "Layers.h"

// A scene that creates a pyramid of boxes to create a very large island
class PyramidScene : public PerformanceTestScene
{
public:
	virtual const char *	GetName() const override
	{
		return "Pyramid";
	}

	virtual void			StartTest(PhysicsSystem &inPhysicsSystem, EMotionQuality inMotionQuality) override
	{
		BodyInterface &bi = inPhysicsSystem.GetBodyInterface();

		// Floor
		bi.CreateAndAddBody(BodyCreationSettings(new BoxShape(Vec3(50.0f, 1.0f, 50.0f), 0.0f), RVec3(Vec3(0.0f, -1.0f, 0.0f)), Quat::sIdentity(), EMotionType::Static, Layers::NON_MOVING), EActivation::DontActivate);

		const float cBoxSize = 2.0f;
		const float cBoxSeparation = 0.5f;
		const float cHalfBoxSize = 0.5f * cBoxSize;
		const int cPyramidHeight = 15;

		RefConst<Shape> box_shape = new BoxShape(Vec3::sReplicate(cHalfBoxSize), 0.0f); // No convex radius to force more collisions

		// Pyramid
		for (int i = 0; i < cPyramidHeight; ++i)
			for (int j = i / 2; j < cPyramidHeight - (i + 1) / 2; ++j)
				for (int k = i / 2; k < cPyramidHeight - (i + 1) / 2; ++k)
				{
					RVec3 position(Real(-cPyramidHeight + cBoxSize * j + (i & 1? cHalfBoxSize : 0.0f)), Real(1.0f + (cBoxSize + cBoxSeparation) * i), Real(-cPyramidHeight + cBoxSize * k + (i & 1? cHalfBoxSize : 0.0f)));
					BodyCreationSettings settings(box_shape, position, Quat::sIdentity(), EMotionType::Dynamic, Layers::MOVING);
					settings.mAllowSleeping = false; // No sleeping to force the large island to stay awake
					bi.CreateAndAddBody(settings, EActivation::Activate);
				}
	}
};


Ref<SoftBodySharedSettings> CreateSphere(float inRadius = 1.0f, uint inNumTheta = 10, uint inNumPhi = 20, SoftBodySharedSettings::EBendType inBendType = SoftBodySharedSettings::EBendType::None, const SoftBodySharedSettings::VertexAttributes &inVertexAttributes = { 1.0e-4f, 1.0e-4f, 1.0e-3f })
{
	// Create settings
	SoftBodySharedSettings *settings = new SoftBodySharedSettings;

	// Create vertices
	// NOTE: This is not how you should create a soft body sphere, we explicitly use polar coordinates to make the vertices unevenly distributed.
	// Doing it this way tests the pressure algorithm as it receives non-uniform triangles. Better is to use uniform triangles,
	// see the use of DebugRenderer::Create8thSphere for an example.
	SoftBodySharedSettings::Vertex v;
	(inRadius * Vec3::sUnitSpherical(0, 0)).StoreFloat3(&v.mPosition);
	settings->mVertices.push_back(v);
	(inRadius * Vec3::sUnitSpherical(JPH_PI, 0)).StoreFloat3(&v.mPosition);
	settings->mVertices.push_back(v);
	for (uint theta = 1; theta < inNumTheta - 1; ++theta)
		for (uint phi = 0; phi < inNumPhi; ++phi)
		{
			(inRadius * Vec3::sUnitSpherical(JPH_PI * theta / (inNumTheta - 1), 2.0f * JPH_PI * phi / inNumPhi)).StoreFloat3(&v.mPosition);
			settings->mVertices.push_back(v);
		}

	// Function to get the vertex index of a point on the sphere
	auto vertex_index = [inNumTheta, inNumPhi](uint inTheta, uint inPhi) -> uint
	{
		if (inTheta == 0)
			return 0;
		else if (inTheta == inNumTheta - 1)
			return 1;
		else
			return 2 + (inTheta - 1) * inNumPhi + inPhi % inNumPhi;
	};

	// Create faces
	SoftBodySharedSettings::Face f;
	for (uint phi = 0; phi < inNumPhi; ++phi)
	{
		for (uint theta = 0; theta < inNumTheta - 2; ++theta)
		{
			f.mVertex[0] = vertex_index(theta, phi);
			f.mVertex[1] = vertex_index(theta + 1, phi);
			f.mVertex[2] = vertex_index(theta + 1, phi + 1);
			settings->AddFace(f);

			if (theta > 0)
			{
				f.mVertex[1] = vertex_index(theta + 1, phi + 1);
				f.mVertex[2] = vertex_index(theta, phi + 1);
				settings->AddFace(f);
			}
		}

		f.mVertex[0] = vertex_index(inNumTheta - 2, phi + 1);
		f.mVertex[1] = vertex_index(inNumTheta - 2, phi);
		f.mVertex[2] = vertex_index(inNumTheta - 1, 0);
		settings->AddFace(f);
	}

	// Create constraints
	settings->CreateConstraints(&inVertexAttributes, 1, inBendType);

	// Optimize the settings
	settings->Optimize();

	return settings;
}

class SoftBodyScene : public PerformanceTestScene
{
public:
	virtual const char *	GetName() const override
	{
		return "SoftBody";
	}
	virtual void			StartTest(PhysicsSystem &inPhysicsSystem, EMotionQuality inMotionQuality) override
	{
		BodyInterface &bi = inPhysicsSystem.GetBodyInterface();

		// Floor
		bi.CreateAndAddBody(BodyCreationSettings(new BoxShape(Vec3(100.0f, 1.0f, 100.0f), 0.0f), RVec3(Vec3(0.0f, -1.0f, 0.0f)), Quat::sIdentity(), EMotionType::Static, Layers::NON_MOVING), EActivation::DontActivate);

		const float cBoxSize = 2.0f;
		
		const float cHalfBoxSize = 0.5f * cBoxSize;
		

		RefConst<Shape> box_shape = new BoxShape(Vec3::sReplicate(cHalfBoxSize), 0.0f); // No convex radius to force more collisions


		SoftBodyCreationSettings sphere(CreateSphere(), RVec3::sZero(), Quat::sIdentity(), Layers::MOVING);
		sphere.mPressure = 2000.0f;

		// Box settings
		BodyCreationSettings box(new BoxShape(Vec3::sOne()), RVec3::sZero(), Quat::sIdentity(), EMotionType::Dynamic, Layers::MOVING);
		box.mOverrideMassProperties = EOverrideMassProperties::CalculateInertia;
		box.mMassPropertiesOverride.mMass = 100.0f;

		for (int x = 0; x <= 15; ++x)
			for (int z = 0; z <= 15; ++z)
			{
				sphere.mPosition = RVec3(-20.0_r + 4.0_r * x, 5.0_r, -20.0_r + 4.0_r * z);
				bi.CreateAndAddSoftBody(sphere, EActivation::Activate);

				box.mPosition = sphere.mPosition + RVec3(0, 4, 0);
				bi.CreateAndAddBody(box, EActivation::Activate);
			}
	}
	};
