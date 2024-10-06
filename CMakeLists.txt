cmake_minimum_required(VERSION 3.10)
project(conv2d_project)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add all source files
set(SOURCE_FILES main.cpp im2col.cpp conv2d.cpp conv1d.cpp)

# Add the executable
add_executable(conv2d_executable ${SOURCE_FILES})

# Set compiler flags for optimization and AVX2 support
if(MSVC)
    target_compile_options(conv2d_executable PRIVATE /O2 /arch:AVX2)
else()
    target_compile_options(conv2d_executable PRIVATE -O3 -mavx2 -mfma)
endif()

# Enable OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(conv2d_executable PUBLIC OpenMP::OpenMP_CXX)
endif()

# Find BLAS library
find_package(BLAS REQUIRED)
if(BLAS_FOUND)
    target_link_libraries(conv2d_executable PUBLIC ${BLAS_LIBRARIES})
else()
    message(FATAL_ERROR "BLAS library not found")
endif()

# Include directories if necessary (e.g., for BLAS headers)
include_directories(${BLAS_INCLUDE_DIRS})

