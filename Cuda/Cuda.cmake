# Find all CUDA source files
# set CUDA root directory

add_library(${PROJECT_NAME} STATIC)

set(CUDA_ROOT ${PHYSICS_REPO_ROOT}/Cuda)

# Find all CUDA source files
file(GLOB_RECURSE CUDA_SRC_FILES
    ${CUDA_ROOT}/*.cu
    ${CUDA_ROOT}/*.cuh
)

# Add all CUDA source files to the project
foreach(CUDA_SRC_FILE ${CUDA_SRC_FILES})

    message(STATUS "Adding CUDA source file: ${CUDA_SRC_FILE}")

    # Get the file name
    get_filename_component(CUDA_SRC_FILE_NAME ${CUDA_SRC_FILE} NAME)
    # Add the file to the project
    target_sources(${PROJECT_NAME} PRIVATE ${CUDA_SRC_FILE})
endforeach()


