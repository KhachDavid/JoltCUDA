

// void BoxShape::CollideSoftBodyVertices(Mat44Arg inCenterOfMassTransform, Vec3Arg inScale, const CollideSoftBodyVertexIterator &inVertices, uint inNumVertices, int inCollidingShapeIndex) const
// {
// 	// Print the inCenterOfMassTransform and what 
// 	printf("Counter: %d\n", counter);
// 	for (int i = 0; i < 4; i++)
// 	{
// 		// print the first row
// 		printf("%f ", inCenterOfMassTransform.);
// 	}
// 	counter++;
// 	Mat44 inverse_transform = inCenterOfMassTransform.InversedRotationTranslation();
// 	Vec3 half_extent = inScale.Abs() * mHalfExtent;

// 	for (CollideSoftBodyVertexIterator v = inVertices, sbv_end = inVertices + inNumVertices; v != sbv_end; ++v)
// 		if (v.GetInvMass() > 0.0f)
// 		{
// 			// Convert to local space
// 			Vec3 local_pos = inverse_transform * v.GetPosition();

// 			// Clamp point to inside box
// 			Vec3 clamped_point = Vec3::sMax(Vec3::sMin(local_pos, half_extent), -half_extent);

// 			// Test if point was inside
// 			if (clamped_point == local_pos)
// 			{
// 				// Calculate closest distance to surface
// 				Vec3 delta = half_extent - local_pos.Abs();
// 				int index = delta.GetLowestComponentIndex();
// 				float penetration = delta[index];
// 				if (v.UpdatePenetration(penetration))
// 				{
// 					// Calculate contact point and normal
// 					Vec3 possible_normals[] = { Vec3::sAxisX(), Vec3::sAxisY(), Vec3::sAxisZ() };
// 					Vec3 normal = local_pos.GetSign() * possible_normals[index];
// 					Vec3 point = normal * half_extent;

// 					// Store collision
// 					v.SetCollision(Plane::sFromPointAndNormal(point, normal).GetTransformed(inCenterOfMassTransform), inCollidingShapeIndex);
// 				}
// 			}
// 			else
// 			{
// 				// Calculate normal
// 				Vec3 normal = local_pos - clamped_point;
// 				float normal_length = normal.Length();

// 				// Penetration will be negative since we're not penetrating
// 				float penetration = -normal_length;
// 				if (v.UpdatePenetration(penetration))
// 				{
// 					normal /= normal_length;

// 					// Store collision
// 					v.SetCollision(Plane::sFromPointAndNormal(clamped_point, normal).GetTransformed(inCenterOfMassTransform), inCollidingShapeIndex);
// 				}
// 			}
// 		}
// }

// write a cude hello world program and add all the necessary files to the project

