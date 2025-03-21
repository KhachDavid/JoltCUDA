# Root
set(JOLT_VIEWER_ROOT ${PHYSICS_REPO_ROOT}/JoltViewer)

# Source files
set(JOLT_VIEWER_SRC_FILES
	${JOLT_VIEWER_ROOT}/JoltViewer.cmake
	${JOLT_VIEWER_ROOT}/JoltViewer.cpp
	${JOLT_VIEWER_ROOT}/JoltViewer.h
)


# Ensure CUDA support
enable_language(CUDA)
set(JOLT_PHYSICS_ROOT ${PHYSICS_REPO_ROOT}/Jolt)


set(JOLT_PHYSICS_CUDA_FILES
    ${JOLT_PHYSICS_ROOT}/Cuda/Hello.cu
	${JOLT_PHYSICS_ROOT}/Cuda/Hello.h
	${JOLT_PHYSICS_ROOT}/Cuda/CollideSoftBodyVertices.cu
	${JOLT_PHYSICS_ROOT}/Cuda/CollideSoftBodyVertices.h
)



# Group source files
source_group(TREE ${JOLT_VIEWER_ROOT} FILES ${JOLT_VIEWER_SRC_FILES})

# Create JoltViewer executable
if ("${CMAKE_SYSTEM_NAME}" MATCHES "Darwin")
	# Icon
	set(JPH_ICON "${CMAKE_CURRENT_SOURCE_DIR}/macOS/icon.icns")
	set_source_files_properties(${JPH_ICON} PROPERTIES MACOSX_PACKAGE_LOCATION "Resources")

	add_executable(JoltViewer MACOSX_BUNDLE ${JOLT_VIEWER_SRC_FILES} ${TEST_FRAMEWORK_ASSETS} ${JPH_ICON})
	set_property(TARGET JoltViewer PROPERTY MACOSX_BUNDLE_INFO_PLIST "${CMAKE_CURRENT_SOURCE_DIR}/iOS/JoltViewerInfo.plist")
	set_property(TARGET JoltViewer PROPERTY XCODE_ATTRIBUTE_PRODUCT_BUNDLE_IDENTIFIER "com.joltphysics.joltviewer")
else()
	add_executable(JoltViewer ${JOLT_VIEWER_SRC_FILES})
endif()
target_include_directories(JoltViewer PUBLIC ${JOLT_VIEWER_ROOT})
target_link_libraries(JoltViewer LINK_PUBLIC TestFramework)

# Set the correct working directory
set_property(TARGET JoltViewer PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${PHYSICS_REPO_ROOT}")


set_target_properties(JoltViewer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(JoltViewer PRIVATE cuda cudart)
